"""
3단계: 웹 페이지에서 "촬영" 버튼으로 현재 프레임을 저장하기.

2단계(step2_server.py)의 Camera와 MJPEG 스트림을 그대로 재사용하고,
촬영(POST /capture)과 저장 이미지 제공(GET /captures/파일명) 기능을 더한다.
"""

import atexit
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, send_from_directory

import config
from step2_server import HOST, PORT, JPEG_QUALITY, Camera, generate_frames, get_backend


# 캡처 이미지를 저장할 폴더 (이 파일과 같은 위치의 captures/)
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

# 페이지에 보여줄 촬영 이력 개수
HISTORY_LIMIT = 10


def make_capture_path(capture_dir):
    """
    <summary>현재 시각으로 YYYYMMDD_HHMMSS.jpg 형식의 저장 경로를 만든다.</summary>
    <param name="capture_dir">이미지를 저장할 폴더 (Path)</param>
    <input>현재 시각</input>
    <output>아직 존재하지 않는 저장 경로 (Path)</output>
    <reason>
    시각을 파일명으로 쓰면 이름만 보고도 촬영 순서를 알 수 있고 정렬도 쉽다.
    같은 초에 두 번 누르면 이름이 겹쳐 덮어쓰게 되므로 _1, _2 번호를 붙여 피한다.
    </reason>
    """
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = capture_dir / f"{stem}.jpg"
    count = 1

    while path.exists():
        path = capture_dir / f"{stem}_{count}.jpg"
        count += 1

    return path


def list_recent_captures(capture_dir, limit):
    """
    <summary>저장된 캡처 파일 이름을 최신순으로 limit개까지 돌려준다.</summary>
    <param name="capture_dir">캡처 폴더 (Path)</param>
    <param name="limit">돌려줄 최대 개수</param>
    <input>captures/ 폴더의 jpg 파일들</input>
    <output>파일 이름 리스트 (list[str]), 최신 파일이 앞쪽</output>
    <reason>
    페이지를 새로고침해도 이전 촬영 이력이 보이도록 서버가 폴더를 직접 읽어 알려준다.
    수정 시각 기준으로 정렬해 같은 초의 _1, _2 파일도 순서가 맞게 한다.
    </reason>
    """
    files = sorted(capture_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)

    return [p.name for p in files[:limit]]


def create_app(camera, capture_dir):
    """
    <summary>스트림 + 촬영 기능을 가진 Flask 앱을 만든다.</summary>
    <param name="camera">모든 요청이 공유할 Camera 객체</param>
    <param name="capture_dir">캡처 이미지를 저장할 폴더 (Path)</param>
    <input>Camera 객체, 저장 폴더</input>
    <output>설정이 끝난 Flask 앱</output>
    <reason>
    2단계의 경로(/ , /video_feed)에 촬영 관련 경로 두 개를 더한다.
    카메라와 저장 폴더를 인자로 받아 테스트나 다음 단계에서 바꿔 끼우기 쉽게 한다.
    </reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>스트림, 촬영 버튼, 촬영 이력을 보여주는 페이지를 돌려준다.</summary>
        <input>브라우저의 GET / 요청</input>
        <output>templates/capture.html을 렌더링한 HTML</output>
        <reason>첫 화면에 기존 촬영 이력을 서버에서 바로 채워 넣어 추가 요청 없이 보여준다.</reason>
        """
        return render_template(
            "capture.html",
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            captures=list_recent_captures(capture_dir, HISTORY_LIMIT),
            history_limit=HISTORY_LIMIT,
        )

    @app.route("/video_feed")
    def video_feed():
        """
        <summary>2단계와 같은 MJPEG 스트림 응답을 돌려준다.</summary>
        <input>브라우저의 GET /video_feed 요청</input>
        <output>multipart/x-mixed-replace 스트리밍 응답</output>
        <reason>스트림 생성 로직은 step2의 generate_frames를 그대로 재사용한다.</reason>
        """
        return Response(
            generate_frames(camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/capture", methods=["POST"])
    def capture():
        """
        <summary>현재 최신 프레임을 jpg로 저장하고 파일명을 JSON으로 돌려준다.</summary>
        <input>브라우저의 POST /capture 요청 (본문 없음)</input>
        <output>성공: {"filename": "..."} (201), 실패: {"error": "..."} (503 또는 500)</output>
        <reason>
        서버의 상태(파일)를 새로 만드는 동작이므로 GET이 아닌 POST를 쓴다.
        스트림이 이미 들고 있는 최신 프레임을 쓰기 때문에 카메라를 다시 열 필요가 없다.
        cv2.imwrite는 Windows에서 한글이 섞인 경로("작업 파일" 등)에 저장하지 못하므로
        imencode로 JPEG 바이트를 만든 뒤 파이썬 write_bytes로 저장한다.
        </reason>
        """
        frame = camera.get_frame()

        if frame is None:
            return jsonify(error="아직 카메라 프레임이 없습니다. 잠시 후 다시 시도하세요."), 503

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if not ok:
            return jsonify(error="이미지를 JPEG로 변환하지 못했습니다."), 500

        path = make_capture_path(capture_dir)

        try:
            path.write_bytes(jpeg.tobytes())
        except OSError as error:
            print(f"[오류] 캡처 저장 실패: {error}")
            return jsonify(error="이미지를 저장하지 못했습니다. 폴더 권한을 확인하세요."), 500

        print(f"[정보] 캡처 저장: {path}")

        return jsonify(filename=path.name), 201

    @app.route("/captures/<path:filename>")
    def captured_image(filename):
        """
        <summary>captures/ 폴더 안의 저장 이미지를 돌려준다.</summary>
        <param name="filename">요청한 파일 이름</param>
        <input>브라우저의 GET /captures/파일명 요청</input>
        <output>jpg 이미지 응답, 폴더 밖을 가리키거나 없는 파일이면 404</output>
        <reason>
        open(capture_dir / filename)처럼 직접 합치면 "../config.py" 같은 요청으로
        폴더 밖 파일이 새어 나갈 수 있다(경로 조작 공격).
        send_from_directory는 내부에서 safe_join으로 폴더 밖 경로를 막아 준다.
        </reason>
        """
        return send_from_directory(capture_dir, filename)

    return app


def main():
    """
    <summary>카메라를 열고 캡처 기능이 있는 Flask 서버를 실행한다.</summary>
    <input>config.py 설정값</input>
    <output>http://127.0.0.1:5000 에서 동작하는 웹 서버, captures/ 폴더의 jpg 파일</output>
    <reason>
    카메라 생성, 종료 처리(atexit), 서버 옵션은 2단계와 같게 유지해
    학생이 "무엇이 추가됐는지"에만 집중하게 한다.
    </reason>
    """
    print(f"[정보] 추론 장치 설정(DEVICE): {config.DEVICE}  (3단계에서는 사용하지 않음)")

    CAPTURE_DIR.mkdir(exist_ok=True)

    backend = get_backend(config.USE_DSHOW)

    try:
        camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, backend)
    except RuntimeError as error:
        print(f"[오류] {error}")
        print("[안내] step1_webcam.py를 실행해 사용 가능한 카메라 번호를 확인하고")
        print("[안내] config.py의 CAMERA_INDEX를 바꿔 보세요.")
        print("[안내] 다른 프로그램(Zoom, Teams, step2 서버 등)이 카메라를 쓰고 있지 않은지도 확인하세요.")
        return

    atexit.register(camera.release)

    app = create_app(camera, CAPTURE_DIR)

    print(f"[안내] 브라우저에서 http://{HOST}:{PORT} 에 접속하세요.")
    print(f"[안내] 캡처 이미지는 {CAPTURE_DIR} 에 저장됩니다.")
    print("[안내] 종료하려면 이 터미널에서 Ctrl+C를 누르세요.")

    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 실행 (step2_server.py가 같은 폴더에 있어야 한다)
#      python step3_capture.py
# 3. 브라우저에서 http://127.0.0.1:5000 접속 → "촬영" 버튼 클릭
# 4. captures/ 폴더에 YYYYMMDD_HHMMSS.jpg 파일이 생기는지 확인한다.
# 5. step2 서버가 켜져 있으면 포트와 카메라가 충돌하므로 먼저 끈다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 동영상 = 연속된 이미지 한 장의 모음
#    - 스트림은 초당 약 30장의 jpg를 이어 보낸 것이고,
#      "촬영"은 그중 지금 이 순간의 한 장을 파일로 꺼내 둔 것뿐이다.
#    - 저장된 jpg의 해상도가 스트림과 같은 640 x 480인지 확인한다.
# 2. 저장된 jpg는 다음 단계 실습 재료
#    - 다양한 물체(사람, 컵, 휴대폰, 책 등)를 여러 장 찍어 둔다.
#    - 다음 단계에서 이 사진들에 YOLO를 적용해 인식 결과를 비교한다.
# 3. fetch와 JSON
#    - 개발자도구(F12) > Network에서 capture 요청의 응답 {"filename": "..."}을 확인한다.
#    - 페이지가 새로고침되지 않고 이미지만 바뀌는 것을 확인한다.
# 4. 경로 조작 방지
#    - http://127.0.0.1:5000/captures/..%2Fconfig.py 에 접속하면 404가 나와야 한다.
# 5. 한글 경로 주의 (Windows)
#    - cv2.imwrite / cv2.imread는 한글이 섞인 경로에서 조용히 실패한다 (False / None 반환).
#    - 그래서 저장은 cv2.imencode + write_bytes 조합을 쓴다.
