"""
2단계: Flask로 웹캠 영상을 브라우저에 스트리밍하기.

카메라는 별도 스레드가 계속 읽고, 최신 프레임 1장만 공유한다.
브라우저는 /video_feed 주소로 MJPEG 스트림을 받아 <img> 태그에 표시한다.
"""

import atexit
import sys
import threading
import time

import cv2
from flask import Flask, Response, render_template

import config


# 서버 주소와 포트 (127.0.0.1은 내 컴퓨터에서만 접속 가능)
HOST = "127.0.0.1"
PORT = 5000

# JPEG 압축 품질 (0~100, 높을수록 화질은 좋고 전송량은 커진다)
JPEG_QUALITY = 80

# 스트림 전송 간격 (초). 카메라 30FPS에 맞춰 약 0.033초.
STREAM_INTERVAL = 1 / 30


def get_backend(use_dshow):
    """
    <summary>카메라를 열 때 사용할 OpenCV 백엔드 번호를 고른다.</summary>
    <param name="use_dshow">True이면 Windows에서 DirectShow 백엔드를 사용한다.</param>
    <input>config.USE_DSHOW 값</input>
    <output>cv2.CAP_DSHOW 또는 cv2.CAP_ANY (int)</output>
    <reason>
    Windows 기본 백엔드는 카메라가 늦게 켜지는 경우가 있어 DirectShow를 선택할 수 있게 한다.
    DirectShow는 Windows 전용이므로 다른 OS에서는 자동 선택(CAP_ANY)을 쓴다.
    </reason>
    """
    if use_dshow and sys.platform == "win32":
        return cv2.CAP_DSHOW

    return cv2.CAP_ANY


class Camera:
    """
    <summary>백그라운드 스레드에서 웹캠을 계속 읽고 최신 프레임 1장을 공유하는 클래스.</summary>
    <param name="index">카메라 번호</param>
    <param name="width">요청할 가로 해상도 (픽셀)</param>
    <param name="height">요청할 세로 해상도 (픽셀)</param>
    <param name="backend">카메라를 열 때 사용할 OpenCV 백엔드</param>
    <input>실제 웹캠 장치</input>
    <output>get_frame()으로 가장 최근 프레임(numpy 배열)을 돌려준다.</output>
    <reason>
    웹캠은 한 번에 한 프로그램(한 VideoCapture)만 안정적으로 열 수 있다.
    요청마다 카메라를 열면 장치 충돌이 나기 때문에,
    카메라는 서버 시작 시 한 번만 열고 모든 요청이 이 객체를 함께 쓴다.
    읽기는 별도 스레드가 담당하므로 브라우저 요청이 카메라 읽기를 기다리지 않는다.
    프레임 변수는 읽기 스레드(쓰기)와 요청 스레드(읽기)가 동시에 접근하므로
    lock으로 보호해 반쯤 바뀐 상태를 읽는 일을 막는다.
    </reason>
    """

    def __init__(self, index, width, height, backend):
        """
        <summary>카메라를 열고 해상도를 요청한 뒤 읽기 스레드를 시작한다.</summary>
        <param name="index">카메라 번호</param>
        <param name="width">요청할 가로 해상도 (픽셀)</param>
        <param name="height">요청할 세로 해상도 (픽셀)</param>
        <param name="backend">카메라를 열 때 사용할 OpenCV 백엔드</param>
        <input>config의 카메라 설정값</input>
        <output>없음. 카메라를 열지 못하면 RuntimeError를 발생시킨다.</output>
        <reason>
        카메라가 열리지 않은 상태로 서버가 뜨면 빈 화면만 보이므로
        생성 시점에 바로 실패를 알려 원인을 빨리 찾게 한다.
        </reason>
        """
        self.cap = cv2.VideoCapture(index, backend)

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"{index}번 카메라를 열 수 없습니다.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        # daemon=True: 메인 프로그램이 끝나면 이 스레드도 함께 끝난다.
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        """
        <summary>읽기 스레드 본체. 카메라에서 프레임을 계속 읽어 self.frame을 교체한다.</summary>
        <input>self.cap에서 읽은 프레임</input>
        <output>없음 (self.frame을 최신 프레임으로 갱신)</output>
        <reason>
        큐에 프레임을 쌓으면 처리보다 읽기가 빠를 때 지연이 계속 늘어난다.
        최신 1장만 덮어쓰면 항상 "지금" 화면을 보여줄 수 있다.
        카메라 자원은 이 스레드만 만지도록 해서 해제도 여기서 한다.
        </reason>
        """
        try:
            while self.running:
                ok, frame = self.cap.read()

                if not ok:
                    print("[오류] 프레임을 읽지 못했습니다. 카메라 연결이 끊겼는지 확인하세요.")
                    self.running = False
                    break

                with self.lock:
                    self.frame = frame

        finally:
            self.cap.release()

    def get_frame(self):
        """
        <summary>가장 최근 프레임의 복사본을 돌려준다.</summary>
        <input>읽기 스레드가 저장한 self.frame</input>
        <output>numpy 배열 (아직 프레임이 없으면 None)</output>
        <reason>
        복사본을 주면 호출한 쪽이 이미지를 수정해도 공유 프레임에 영향이 없다.
        lock은 복사하는 짧은 순간에만 잡아 읽기 스레드를 오래 막지 않는다.
        </reason>
        """
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def release(self):
        """
        <summary>읽기 스레드를 멈추고 카메라 자원이 해제될 때까지 기다린다.</summary>
        <input>없음</input>
        <output>없음</output>
        <reason>
        카메라를 해제하지 않고 종료하면 다음 실행 때 카메라가 "사용 중"으로 남을 수 있다.
        여러 번 호출해도 안전하므로 atexit에 등록해 두고 써도 된다.
        </reason>
        """
        self.running = False

        if self.thread.is_alive():
            self.thread.join(timeout=2)

        print("[정보] 카메라를 해제했습니다.")


def generate_frames(camera):
    """
    <summary>최신 프레임을 JPEG로 압축해 MJPEG 스트림 조각으로 계속 내보내는 제너레이터.</summary>
    <param name="camera">프레임을 공유하는 Camera 객체</param>
    <input>camera.get_frame()으로 얻은 BGR 프레임</input>
    <output>multipart 경계(--frame)와 헤더가 붙은 JPEG 바이트 조각 (bytes)</output>
    <reason>
    MJPEG은 JPEG 이미지를 경계 문자열로 구분해 연달아 보내는 방식이라
    브라우저의 <img> 태그만으로 별도 플레이어 없이 영상을 볼 수 있다.
    yield를 쓰면 응답을 끝내지 않고 조각을 계속 흘려보낼 수 있다.
    </reason>
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while camera.running:
        frame = camera.get_frame()

        if frame is None:
            time.sleep(0.01)
            continue

        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )

        # 같은 프레임을 너무 자주 보내지 않도록 카메라 속도에 맞춰 쉰다.
        time.sleep(STREAM_INTERVAL)


def create_app(camera):
    """
    <summary>Flask 앱을 만들고 페이지(/)와 스트림(/video_feed) 경로를 등록한다.</summary>
    <param name="camera">모든 요청이 공유할 Camera 객체</param>
    <input>Camera 객체</input>
    <output>설정이 끝난 Flask 앱</output>
    <reason>
    카메라를 전역 변수 대신 인자로 넘기면 앱이 어떤 카메라를 쓰는지 한눈에 보이고,
    다음 단계에서 다른 Camera(예: YOLO 추론 포함)로 바꿔 끼우기 쉽다.
    </reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>영상을 보여주는 메인 페이지를 돌려준다.</summary>
        <input>브라우저의 GET / 요청</input>
        <output>templates/index.html을 렌더링한 HTML</output>
        <reason>HTML을 파이썬 코드와 분리해 디자인을 따로 고칠 수 있게 한다.</reason>
        """
        return render_template("index.html", width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT)

    @app.route("/video_feed")
    def video_feed():
        """
        <summary>MJPEG 스트림 응답을 돌려준다.</summary>
        <input>브라우저의 GET /video_feed 요청 (img 태그가 자동으로 보냄)</input>
        <output>mimetype이 multipart/x-mixed-replace인 스트리밍 응답</output>
        <reason>
        x-mixed-replace는 "새 조각이 오면 이전 이미지를 교체하라"는 뜻이라
        이미지가 연속으로 바뀌면서 영상처럼 보인다.
        </reason>
        """
        return Response(
            generate_frames(camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main():
    """
    <summary>카메라를 열고 Flask 서버를 실행한다.</summary>
    <input>config.py 설정값</input>
    <output>http://127.0.0.1:5000 에서 동작하는 웹 서버</output>
    <reason>
    use_reloader=False로 두어야 Flask 개발 서버가 프로세스를 두 번 띄우지 않는다.
    두 번 뜨면 카메라도 두 번 열려 장치 충돌이 난다.
    threaded=True로 여러 요청(페이지, 스트림)을 동시에 처리한다.
    </reason>
    """
    print(f"[정보] 추론 장치 설정(DEVICE): {config.DEVICE}  (2단계에서는 사용하지 않음)")

    backend = get_backend(config.USE_DSHOW)

    try:
        camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, backend)
    except RuntimeError as error:
        print(f"[오류] {error}")
        print("[안내] step1_webcam.py를 실행해 사용 가능한 카메라 번호를 확인하고")
        print("[안내] config.py의 CAMERA_INDEX를 바꿔 보세요.")
        print("[안내] 다른 프로그램(Zoom, Teams, step1 창 등)이 카메라를 쓰고 있지 않은지도 확인하세요.")
        return

    # 서버가 어떤 방식으로 끝나든(Ctrl+C 포함) 카메라를 해제한다.
    atexit.register(camera.release)

    app = create_app(camera)

    print(f"[안내] 브라우저에서 http://{HOST}:{PORT} 에 접속하세요.")
    print("[안내] 종료하려면 이 터미널에서 Ctrl+C를 누르세요.")

    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 실행
#      python step2_server.py
# 3. 브라우저에서 http://127.0.0.1:5000 접속
# 4. 종료: 터미널에서 Ctrl+C → "[정보] 카메라를 해제했습니다." 출력 확인
# 5. step1 창이 열려 있으면 카메라가 충돌하므로 먼저 닫는다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. multipart/x-mixed-replace 응답 구조
#    - 브라우저 개발자도구(F12) > Network 탭에서 video_feed 요청을 클릭한다.
#    - Content-Type이 "multipart/x-mixed-replace; boundary=frame" 인지 확인한다.
#    - 응답이 "끝나지 않고" 계속 받는 중(pending)으로 남아 있는 것을 확인한다.
#    - 한 조각의 구조:
#        --frame
#        Content-Type: image/jpeg
#        (빈 줄)
#        (JPEG 바이트)
# 2. 스레드가 필요한 이유
#    - cap.read()는 다음 프레임이 올 때까지 기다리는(블로킹) 함수다.
#    - 요청 처리 중에 직접 읽으면 요청마다 대기가 생기고, 요청마다 카메라를 열면 장치 충돌이 난다.
#    - 읽기 전용 스레드 하나가 카메라를 독점하고, 요청들은 결과만 가져간다.
# 3. lock이 필요한 이유
#    - self.frame을 읽기 스레드는 쓰고, 요청 스레드는 읽는다.
#    - lock 없이 동시에 접근하면 교체 도중의 값을 읽을 수 있다.
#    - 브라우저 탭을 2~3개 열어도 영상이 모두 나오는지 확인한다 (카메라는 1번만 열림).
# 4. JPEG_QUALITY를 30과 95로 바꿔 화질과 반응 속도 차이를 비교한다.
