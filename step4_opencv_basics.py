"""
4단계: OpenCV 이미지 기초 (읽기, 정보, 색 공간, 좌표계).

3단계에서 저장한 captures/ 폴더의 가장 최근 jpg 1장을 입력으로 쓴다.
캡처 이미지가 없으면 웹캠에서 1장을 찍어 대신 사용한다.
아무 키나 누르면 다음 창으로 넘어가고, 마지막 창에서 키를 누르면 종료한다.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

import config


# 3단계 캡처 이미지가 저장된 폴더
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

# 비교 화면에서 각 이미지의 높이 (3장을 가로로 붙이므로 작게 줄인다)
PANEL_HEIGHT = 320

# 웹캠 폴백 시 버리는 프레임 수 (켜자마자 찍으면 노출이 맞지 않아 어둡다)
WARMUP_FRAMES = 10


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


def find_latest_capture(capture_dir):
    """
    <summary>캡처 폴더에서 가장 최근에 저장된 jpg 경로를 찾는다.</summary>
    <param name="capture_dir">3단계 캡처 폴더 (Path)</param>
    <input>captures/ 폴더의 jpg 파일들</input>
    <output>가장 최근 jpg 경로 (Path), 없으면 None</output>
    <reason>학생이 파일명을 직접 입력하지 않아도 방금 찍은 사진으로 바로 실습하게 한다.</reason>
    """
    if not capture_dir.is_dir():
        return None

    files = sorted(capture_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)

    if not files:
        return None

    return files[-1]


def load_image(path):
    """
    <summary>이미지 파일을 BGR numpy 배열로 읽는다.</summary>
    <param name="path">읽을 이미지 경로 (Path)</param>
    <input>jpg/png 파일</input>
    <output>BGR 이미지 (numpy 배열, uint8), 읽지 못하면 None</output>
    <reason>
    기본은 cv2.imread지만, Windows에서는 경로에 한글이 섞이면
    imread가 오류 없이 None을 돌려준다.
    그때는 numpy로 파일 바이트를 읽고 cv2.imdecode로 해석하는 방법으로 다시 시도한다.
    </reason>
    """
    image = cv2.imread(str(path))

    if image is not None:
        return image

    print("[안내] cv2.imread가 None을 반환했습니다. (Windows 한글 경로 문제일 수 있음)")
    print("[안내] np.fromfile + cv2.imdecode 방식으로 다시 읽습니다.")

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as error:
        print(f"[오류] 파일을 열 수 없습니다: {error}")
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def capture_from_webcam(index, width, height, backend):
    """
    <summary>웹캠에서 이미지 1장을 찍어 돌려준다 (캡처 파일이 없을 때의 폴백).</summary>
    <param name="index">카메라 번호</param>
    <param name="width">요청할 가로 해상도 (픽셀)</param>
    <param name="height">요청할 세로 해상도 (픽셀)</param>
    <param name="backend">카메라를 열 때 사용할 OpenCV 백엔드</param>
    <input>실제 웹캠 장치</input>
    <output>BGR 이미지 (numpy 배열), 실패하면 None</output>
    <reason>
    3단계를 건너뛴 학생도 이 단계를 바로 실습할 수 있게 한다.
    카메라는 켠 직후 자동 노출이 맞지 않으므로 앞쪽 몇 장을 버리고 찍는다.
    </reason>
    """
    cap = cv2.VideoCapture(index, backend)

    try:
        if not cap.isOpened():
            print(f"[오류] {index}번 카메라를 열 수 없습니다.")
            print("[안내] 다른 프로그램(step2/step3 서버, Zoom 등)이 카메라를 쓰고 있지 않은지 확인하세요.")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        frame = None

        for _ in range(WARMUP_FRAMES):
            ok, frame = cap.read()

            if not ok:
                print("[오류] 프레임을 읽지 못했습니다. 카메라 연결이 끊겼는지 확인하세요.")
                return None

        return frame

    finally:
        cap.release()


def print_image_info(image):
    """
    <summary>이미지의 shape, dtype, 픽셀값 범위, 채널별 평균을 출력한다.</summary>
    <param name="image">BGR 이미지 (numpy 배열)</param>
    <input>load_image 또는 capture_from_webcam의 결과</input>
    <output>없음 (콘솔 출력)</output>
    <reason>
    이미지가 "숫자가 담긴 3차원 배열"이라는 것을 숫자로 직접 확인하게 한다.
    채널별 평균을 보면 사진의 색 경향(파란 톤, 붉은 톤)을 숫자로 읽을 수 있다.
    </reason>
    """
    height, width, channels = image.shape
    blue_mean, green_mean, red_mean = image.mean(axis=(0, 1))

    print(f"[정보] shape = {image.shape}  → 높이 {height}, 너비 {width}, 채널 {channels}")
    print(f"[정보] dtype = {image.dtype}")
    print(f"[정보] 픽셀값 범위 = {image.min()} ~ {image.max()}")
    print(f"[정보] 채널별 평균 = B:{blue_mean:.1f} G:{green_mean:.1f} R:{red_mean:.1f}")


def resize_to_height(image, height):
    """
    <summary>가로세로 비율을 유지하면서 지정한 높이로 이미지 크기를 바꾼다.</summary>
    <param name="image">입력 이미지 (numpy 배열)</param>
    <param name="height">목표 높이 (픽셀)</param>
    <input>임의 크기의 이미지</input>
    <output>높이가 height인 이미지</output>
    <reason>cv2.hconcat은 모든 이미지의 높이가 같아야 하므로 붙이기 전에 높이를 맞춘다.</reason>
    """
    scale = height / image.shape[0]
    width = round(image.shape[1] * scale)

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def add_label(image, text):
    """
    <summary>이미지 왼쪽 위에 설명 글자를 그린다.</summary>
    <param name="image">글자를 그릴 3채널 이미지 (numpy 배열, 직접 수정됨)</param>
    <param name="text">표시할 영문 글자</param>
    <input>비교용 패널 이미지</input>
    <output>글자가 그려진 같은 이미지</output>
    <reason>
    세 패널이 무엇인지 화면만 보고 구분하게 한다.
    cv2.putText는 한글을 그리지 못하므로 영문으로 쓴다.
    배경 사각형을 깔아 밝은 사진에서도 글자가 잘 보이게 한다.
    </reason>
    """
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (40, 40, 40), thickness=-1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    return image


def make_color_comparison(image):
    """
    <summary>원본(BGR), RGB로 바꾼 배열, 그레이스케일을 가로로 붙인 비교 이미지를 만든다.</summary>
    <param name="image">BGR 이미지 (numpy 배열)</param>
    <input>원본 BGR 이미지</input>
    <output>세 패널을 hconcat으로 붙인 BGR 이미지</output>
    <reason>
    cv2.imshow는 배열을 항상 BGR로 해석한다.
    그래서 RGB로 바꾼 배열을 그대로 보여주면 빨강과 파랑이 뒤바뀌어 보이고,
    "색 순서를 잘못 맞추면 어떻게 되는지"를 눈으로 확인할 수 있다.
    그레이스케일은 채널이 1개라 hconcat을 위해 3채널로 늘려 채널 수를 맞춘다.
    </reason>
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"[정보] 그레이스케일 shape = {gray.shape}  → 채널 차원이 없어짐")

    # hconcat은 높이, dtype, 채널 수가 모두 같아야 한다.
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    panels = [
        add_label(resize_to_height(image, PANEL_HEIGHT), "Original (BGR)"),
        add_label(resize_to_height(rgb, PANEL_HEIGHT), "RGB shown as BGR"),
        add_label(resize_to_height(gray_3ch, PANEL_HEIGHT), "Grayscale"),
    ]

    return cv2.hconcat(panels)


def demonstrate_coordinates(image):
    """
    <summary>(x, y) 좌표와 배열 인덱스 [y, x]의 순서 차이를 코드와 그림으로 보여준다.</summary>
    <param name="image">BGR 이미지 (numpy 배열)</param>
    <input>원본 BGR 이미지</input>
    <output>점과 좌표 글자가 그려진 이미지 (원본은 바꾸지 않음)</output>
    <reason>
    cv2.circle 같은 그리기 함수는 (x, y) = (가로, 세로) 순서로 받고,
    numpy 배열 인덱스는 [y, x] = [행, 열] 순서로 접근한다.
    두 순서를 섞어 쓰는 실수가 매우 흔해서, 같은 점을 두 방식으로 접근해 결과를 비교한다.
    </reason>
    """
    canvas = image.copy()
    height, width = canvas.shape[:2]

    # 가로로 3/4, 세로로 1/4 지점 → x와 y가 확실히 다른 점을 고른다.
    x = width * 3 // 4
    y = height // 4

    marker_color = (180, 120, 60)

    # 그리기 함수: (x, y) 순서
    cv2.circle(canvas, (x, y), 6, marker_color, thickness=-1)
    cv2.putText(canvas, f"(x={x}, y={y})", (x - 60, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 1, cv2.LINE_AA)

    # 원점 표시
    cv2.putText(canvas, "(0, 0)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 1, cv2.LINE_AA)

    print()
    print("========== 좌표계 비교 ==========")
    print(f"이미지 크기: 너비(width) {width}, 높이(height) {height}")
    print(f"cv2.circle(canvas, (x, y), ...)  →  (x, y) = ({x}, {y})")
    print(f"canvas[y, x] = canvas[{y}, {x}] = {canvas[y, x]}  → 방금 그린 점의 색 (B, G, R)")

    # 순서를 바꿔 [x, y]로 접근하면 다른 위치를 읽거나 범위를 벗어난다.
    if x < height and y < width:
        print(f"canvas[x, y] = canvas[{x}, {y}] = {canvas[x, y]}  → 엉뚱한 위치의 색")
    else:
        print(f"canvas[x, y] = canvas[{x}, {y}]  → IndexError! 행 번호 {x} ≥ 높이 {height} (범위 밖)")

    print("정리: 그리기 함수는 (x, y), 배열 인덱스는 [y, x]")
    print("=================================")

    return canvas


def show_and_wait(title, image):
    """
    <summary>이미지를 창에 띄우고 키 입력을 기다린 뒤 창을 닫는다.</summary>
    <param name="title">창 제목 (영문 권장)</param>
    <param name="image">표시할 이미지 (numpy 배열)</param>
    <input>표시할 이미지, 키보드 입력</input>
    <output>없음</output>
    <reason>
    키를 누를 때까지 멈춰 학생이 결과를 충분히 관찰할 수 있게 한다.
    waitKey(0) 하나만 쓰면 창의 X 버튼으로 닫았을 때 멈춘 채 남거나
    이미 닫힌 창을 destroyWindow 하다가 오류가 난다.
    그래서 짧게 기다리기를 반복하며 "키 입력" 또는 "창이 닫힘"을 모두 확인한다.
    </reason>
    """
    print(f"[안내] '{title}' 창에서 아무 키나 누르면(또는 창을 닫으면) 다음으로 넘어갑니다.")

    cv2.imshow(title, image)

    while True:
        if cv2.waitKey(100) != -1:
            cv2.destroyWindow(title)
            break

        # X 버튼으로 창을 닫으면 VISIBLE 값이 1보다 작아진다.
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break


def main():
    """
    <summary>캡처 이미지(또는 웹캠 1장)를 읽어 정보 출력, 색 공간 비교, 좌표계 시연을 차례로 실행한다.</summary>
    <input>captures/ 폴더의 최신 jpg 또는 웹캠 프레임</input>
    <output>콘솔 출력, 비교 창 2개</output>
    <reason>
    입력 확보 → 정보 확인 → 변환 비교 → 좌표 확인 순서로
    YOLO에 들어가기 전 꼭 알아야 할 이미지 기초를 한 번에 훑는다.
    </reason>
    """
    print(f"[정보] 추론 장치 설정(DEVICE): {config.DEVICE}  (4단계에서는 사용하지 않음)")

    path = find_latest_capture(CAPTURE_DIR)
    image = None

    if path is not None:
        print(f"[정보] 입력 이미지: {path.name}")
        image = load_image(path)

        if image is None:
            print(f"[오류] {path.name} 파일을 이미지로 읽지 못했습니다. 파일이 손상되었을 수 있습니다.")

    if image is None:
        print("[안내] 사용할 캡처 이미지가 없어 웹캠에서 1장을 찍습니다.")
        backend = get_backend(config.USE_DSHOW)
        image = capture_from_webcam(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, backend)

    if image is None:
        print("[오류] 입력 이미지를 준비하지 못해 종료합니다.")
        print("[안내] step3_capture.py로 사진을 찍거나 카메라 연결을 확인하세요.")
        return

    print_image_info(image)

    try:
        comparison = make_color_comparison(image)
        show_and_wait("Step4 - Color (BGR / RGB / Gray)", comparison)

        coordinates = demonstrate_coordinates(image)
        show_and_wait("Step4 - Coordinates (x, y) vs [y, x]", coordinates)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. (권장) step3_capture.py로 사진을 몇 장 찍어 둔다.
# 3. 실행
#      python step4_opencv_basics.py
# 4. 창이 뜨면 관찰 후 아무 키나 눌러 다음 창으로 넘어간다.
# 5. captures/ 폴더가 비어 있으면 웹캠에서 1장을 자동으로 찍는다.
#    (step2/step3 서버가 켜져 있으면 카메라가 충돌하므로 먼저 끈다.)
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 이미지 정보
#    - shape는 (높이, 너비, 3), dtype은 uint8, 픽셀값은 0~255 범위 안에 있다.
#    - 그레이스케일은 shape가 (높이, 너비)로 채널 차원이 사라진다.
# 2. OpenCV가 BGR을 쓰는 이유
#    - OpenCV가 처음 만들어질 당시(2000년 전후) 카메라 장비와 Windows 이미지 포맷(BMP)이
#      BGR 순서를 흔히 썼고, 그 관례가 호환성 때문에 지금까지 유지되고 있다.
#    - 반면 matplotlib, PIL, 대부분의 딥러닝 모델은 RGB를 기준으로 한다.
# 3. 왜 YOLO 입력 전에 색 순서가 중요한가
#    - 가운데 패널("RGB shown as BGR")에서 피부색이 파랗게 보이는 것을 확인한다.
#    - 모델은 학습한 색 순서와 다른 입력을 받으면, 사람 눈에 이 패널처럼 보이는
#      "색이 뒤바뀐 사진"을 보게 되어 인식 정확도가 떨어진다.
#    - ultralytics YOLO는 numpy 배열을 BGR(OpenCV 형식)로 가정하고 내부에서 RGB로 바꾼다.
#      따라서 cv2로 읽은 프레임은 "그대로" 넣고, 직접 RGB로 바꿔 넣으면 오히려 틀린다.
# 4. 좌표계
#    - 콘솔에서 canvas[y, x]는 방금 그린 점의 색이고, canvas[x, y]는 다른 값(또는 IndexError)인지 확인한다.
#    - 그리기 함수는 (x, y), 배열 인덱스는 [y, x]라는 규칙을 기억한다.
