"""
5단계: 웹캠 실시간 영상에 OpenCV 처리 모드를 키보드로 바꿔 적용하기.

1: 원본 / 2: 그레이 / 3: 가우시안 블러 / 4: Canny 엣지 / 5: 이진화 / 6: HSV 색상 마스크
s: 현재 처리 결과 저장 / q: 종료
"""

import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

import config


# 처리 결과를 저장할 폴더 (captures/는 4단계 입력용이라 따로 둔다)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# 메인 창과 HSV 트랙바 창 이름
WINDOW_NAME = "Step5 - OpenCV Practice"
HSV_WINDOW_NAME = "HSV Range"

# FPS 이동평균에 사용할 최근 프레임 수
FPS_WINDOW = 30

# 각 모드의 파라미터
BLUR_KERNEL = (21, 21)
CANNY_LOW = 100
CANNY_HIGH = 200
THRESHOLD_VALUE = 127

# 화면 표시 색 (BGR, 채도 낮은 톤)
PANEL_COLOR = (40, 40, 40)
TEXT_COLOR = (235, 235, 235)
ACCENT_COLOR = (150, 200, 120)


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


class FpsCounter:
    """
    <summary>최근 N프레임의 시간 간격으로 FPS 이동평균을 계산한다.</summary>
    <param name="window">평균에 사용할 최근 프레임 수</param>
    <input>매 프레임 tick() 호출</input>
    <output>tick()이 현재 FPS(float)를 돌려준다.</output>
    <reason>
    한 프레임 간격만으로 FPS를 계산하면 값이 매 프레임 크게 흔들려 읽기 어렵다.
    최근 N프레임의 평균을 쓰면 숫자가 안정되어 모드별 차이를 비교하기 쉽다.
    time.perf_counter는 짧은 시간 측정에 가장 정밀한 시계라서 사용한다.
    </reason>
    """

    def __init__(self, window):
        """
        <summary>시각 기록용 큐를 만든다.</summary>
        <param name="window">평균에 사용할 최근 프레임 수</param>
        <input>없음</input>
        <output>없음</output>
        <reason>deque(maxlen)은 오래된 값을 자동으로 버려 이동평균 구현이 간단하다.</reason>
        """
        self.times = deque(maxlen=window)

    def tick(self):
        """
        <summary>현재 시각을 기록하고 FPS 이동평균을 돌려준다.</summary>
        <input>없음 (호출 시점의 perf_counter 값)</input>
        <output>FPS (float), 기록이 2개 미만이면 0.0</output>
        <reason>(기록 개수 - 1)개의 간격이 걸린 전체 시간으로 나누면 평균 FPS가 된다.</reason>
        """
        self.times.append(time.perf_counter())

        if len(self.times) < 2:
            return 0.0

        elapsed = self.times[-1] - self.times[0]

        return (len(self.times) - 1) / elapsed


def to_bgr(single_channel):
    """
    <summary>1채널(흑백) 이미지를 3채널 BGR 이미지로 바꾼다.</summary>
    <param name="single_channel">그레이, 엣지, 이진화 결과 (2차원 배열)</param>
    <input>shape가 (높이, 너비)인 이미지</input>
    <output>shape가 (높이, 너비, 3)인 이미지</output>
    <reason>
    1채널 이미지 위에는 색 있는 글자나 사각형을 그릴 수 없다.
    모든 모드의 결과를 3채널로 통일해 같은 오버레이 함수를 쓰게 한다.
    </reason>
    """
    return cv2.cvtColor(single_channel, cv2.COLOR_GRAY2BGR)


def mode_original(frame):
    """
    <summary>원본 프레임을 그대로 돌려준다.</summary>
    <param name="frame">BGR 프레임</param>
    <input>웹캠 프레임</input>
    <output>같은 프레임</output>
    <reason>다른 모드의 FPS와 비교할 기준선(처리 없음)이 필요하다.</reason>
    """
    return frame


def mode_gray(frame):
    """
    <summary>그레이스케일로 변환한다.</summary>
    <param name="frame">BGR 프레임</param>
    <input>웹캠 프레임</input>
    <output>그레이스케일을 3채널로 늘린 이미지</output>
    <reason>대부분의 고전 영상처리(엣지, 이진화)는 밝기 정보만 쓰므로 첫 단계로 흑백 변환을 익힌다.</reason>
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return to_bgr(gray)


def mode_blur(frame):
    """
    <summary>가우시안 블러를 적용한다.</summary>
    <param name="frame">BGR 프레임</param>
    <input>웹캠 프레임</input>
    <output>흐려진 BGR 이미지</output>
    <reason>
    블러는 노이즈를 줄이는 기본 전처리다.
    커널이 클수록 픽셀마다 계산할 이웃이 많아져 연산량이 늘어나는 것을 FPS로 확인할 수 있다.
    </reason>
    """
    return cv2.GaussianBlur(frame, BLUR_KERNEL, 0)


def mode_canny(frame):
    """
    <summary>Canny 알고리즘으로 엣지(윤곽선)를 검출한다.</summary>
    <param name="frame">BGR 프레임</param>
    <input>웹캠 프레임</input>
    <output>엣지를 흰색으로 표시한 3채널 이미지</output>
    <reason>
    그레이 변환 → 블러 → 엣지 검출로 여러 단계를 거치므로 연산량이 큰 편이다.
    블러를 먼저 하면 노이즈가 엣지로 잘못 잡히는 것을 줄일 수 있다.
    </reason>
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    return to_bgr(edges)


def mode_threshold(frame):
    """
    <summary>밝기 기준값으로 흑과 백 두 가지로 나눈다(이진화).</summary>
    <param name="frame">BGR 프레임</param>
    <input>웹캠 프레임</input>
    <output>이진화 결과를 3채널로 늘린 이미지</output>
    <reason>
    기준값(THRESHOLD_VALUE)보다 밝으면 255, 어두우면 0으로 만든다.
    조명에 따라 결과가 크게 바뀌는 것을 보며 "고정 규칙"의 한계를 체감하게 한다.
    </reason>
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

    return to_bgr(binary)


def create_hsv_trackbars():
    """
    <summary>H/S/V 최솟값·최댓값을 조절하는 트랙바 창을 만든다.</summary>
    <input>없음</input>
    <output>없음 (HSV_WINDOW_NAME 창 생성)</output>
    <reason>
    색 범위를 코드 수정 없이 실시간으로 바꿔 보며 HSV의 의미를 익히게 한다.
    OpenCV의 H는 0~179(360도를 절반으로), S와 V는 0~255 범위다.
    </reason>
    """
    cv2.namedWindow(HSV_WINDOW_NAME)
    cv2.resizeWindow(HSV_WINDOW_NAME, 400, 260)

    # 기본값: 색이 진하고(채도 60 이상) 너무 어둡지 않은(명도 60 이상) 영역
    cv2.createTrackbar("H min", HSV_WINDOW_NAME, 0, 179, lambda _: None)
    cv2.createTrackbar("H max", HSV_WINDOW_NAME, 179, 179, lambda _: None)
    cv2.createTrackbar("S min", HSV_WINDOW_NAME, 60, 255, lambda _: None)
    cv2.createTrackbar("S max", HSV_WINDOW_NAME, 255, 255, lambda _: None)
    cv2.createTrackbar("V min", HSV_WINDOW_NAME, 60, 255, lambda _: None)
    cv2.createTrackbar("V max", HSV_WINDOW_NAME, 255, 255, lambda _: None)


def is_window_open(name):
    """
    <summary>이름으로 지정한 OpenCV 창이 지금 열려 있는지 확인한다.</summary>
    <param name="name">창 이름</param>
    <input>OpenCV 창 상태</input>
    <output>열려 있으면 True</output>
    <reason>
    사용자가 X 버튼으로 창을 닫으면 이후 트랙바 읽기나 창 닫기에서 오류가 난다.
    창이 없을 때 getWindowProperty가 오류를 내는 환경도 있어 함께 처리한다.
    </reason>
    """
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def mode_hsv_mask(frame):
    """
    <summary>트랙바로 정한 HSV 범위 안의 색만 남기고 나머지는 검게 지운다.</summary>
    <param name="frame">BGR 프레임</param>
    <input>웹캠 프레임, HSV 트랙바 값</input>
    <output>범위 안의 색만 남은 BGR 이미지</output>
    <reason>
    BGR은 밝기가 바뀌면 세 값이 모두 바뀌어 "파란색"처럼 색을 고르기 어렵다.
    HSV는 색상(H)과 밝기(V)가 분리되어 있어 조명이 바뀌어도 색 범위를 잡기 쉽다.
    트랙바 창을 사용자가 닫았으면 다시 만들어 오류 없이 계속 동작하게 한다.
    </reason>
    """
    if not is_window_open(HSV_WINDOW_NAME):
        create_hsv_trackbars()

    lower = (
        cv2.getTrackbarPos("H min", HSV_WINDOW_NAME),
        cv2.getTrackbarPos("S min", HSV_WINDOW_NAME),
        cv2.getTrackbarPos("V min", HSV_WINDOW_NAME),
    )
    upper = (
        cv2.getTrackbarPos("H max", HSV_WINDOW_NAME),
        cv2.getTrackbarPos("S max", HSV_WINDOW_NAME),
        cv2.getTrackbarPos("V max", HSV_WINDOW_NAME),
    )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    return cv2.bitwise_and(frame, frame, mask=mask)


# 키 → (화면에 표시할 모드 이름, 처리 함수)
MODES = {
    ord("1"): ("1 Original", mode_original),
    ord("2"): ("2 Gray", mode_gray),
    ord("3"): ("3 Gaussian Blur", mode_blur),
    ord("4"): ("4 Canny Edge", mode_canny),
    ord("5"): ("5 Threshold", mode_threshold),
    ord("6"): ("6 HSV Mask", mode_hsv_mask),
}


def draw_overlay(image, mode_name, fps, process_ms):
    """
    <summary>화면 위쪽에 현재 모드, FPS, 처리 시간을, 아래쪽에 키 안내를 그린다.</summary>
    <param name="image">3채널 결과 이미지 (직접 수정됨)</param>
    <param name="mode_name">현재 모드 이름 (영문)</param>
    <param name="fps">이동평균 FPS</param>
    <param name="process_ms">이번 프레임의 처리 함수 실행 시간 (밀리초)</param>
    <input>모드 처리 결과 이미지</input>
    <output>정보가 그려진 같은 이미지</output>
    <reason>
    cv2.rectangle(배경 상자)과 cv2.putText(글자)는
    STEP 6~7에서 YOLO가 찾은 객체의 박스와 "클래스명 신뢰도" 라벨을 그릴 때 그대로 쓰인다.
    여기서 좌표 (x, y), 색 (B, G, R), 두께 -1(채우기) 사용법을 미리 익힌다.
    FPS는 카메라 속도(약 30)에 막혀 차이가 작게 보일 수 있어,
    처리 함수만의 시간(ms)도 함께 보여 연산량 차이를 정확히 비교하게 한다.
    </reason>
    """
    height, width = image.shape[:2]

    # 위쪽 정보 상자: 두께 -1은 사각형 내부를 채운다.
    cv2.rectangle(image, (0, 0), (width, 36), PANEL_COLOR, thickness=-1)
    cv2.putText(image, mode_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_COLOR, 2, cv2.LINE_AA)

    info = f"FPS {fps:5.1f} | proc {process_ms:5.1f} ms"
    cv2.putText(image, info, (width - 290, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)

    # 아래쪽 키 안내
    cv2.rectangle(image, (0, height - 28), (width, height), PANEL_COLOR, thickness=-1)
    cv2.putText(image, "1-6: mode   s: save   q: quit", (10, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return image


def save_result(image, mode_name, output_dir):
    """
    <summary>현재 처리 결과를 outputs/ 폴더에 jpg로 저장한다.</summary>
    <param name="image">저장할 이미지 (오버레이를 그리기 전의 처리 결과)</param>
    <param name="mode_name">파일 이름에 넣을 모드 이름</param>
    <param name="output_dir">저장 폴더 (Path)</param>
    <input>처리 결과 이미지</input>
    <output>저장된 경로 (Path), 실패하면 None</output>
    <reason>
    글자를 그리기 전의 결과를 저장해 이후 실습 재료로 깨끗하게 쓸 수 있게 한다.
    Windows에서 cv2.imwrite는 한글 경로에 저장하지 못하므로 imencode + write_bytes를 쓴다.
    </reason>
    """
    output_dir.mkdir(exist_ok=True)

    safe_mode = mode_name.split(" ", 1)[1].replace(" ", "_").lower()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = output_dir / f"step5_{safe_mode}_{stamp}.jpg"

    ok, jpeg = cv2.imencode(".jpg", image)

    if not ok:
        return None

    try:
        path.write_bytes(jpeg.tobytes())
    except OSError as error:
        print(f"[오류] 저장 실패: {error}")
        return None

    return path


def main():
    """
    <summary>웹캠을 열고 키보드로 고른 처리 모드를 실시간으로 적용해 보여준다.</summary>
    <input>config.py 설정값, 키보드 입력(1~6, s, q)</input>
    <output>처리 결과 창, s 키를 누를 때마다 outputs/ 폴더의 jpg 파일</output>
    <reason>
    "읽기 → 처리 → 그리기 → 표시"라는 실시간 루프 구조를 유지한 채
    처리 부분만 바꿔 끼우는 형태로 만들어, STEP 7에서 처리 자리에 YOLO를 넣으면 되게 한다.
    </reason>
    """
    print(f"[정보] 추론 장치 설정(DEVICE): {config.DEVICE}  (5단계에서는 사용하지 않음)")

    backend = get_backend(config.USE_DSHOW)
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)

    if not cap.isOpened():
        cap.release()
        print(f"[오류] {config.CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        print("[안내] step1_webcam.py로 사용 가능한 카메라 번호를 확인하고 config.py의 CAMERA_INDEX를 바꿔 보세요.")
        print("[안내] 다른 프로그램(step2/step3 서버, Zoom 등)이 카메라를 쓰고 있지 않은지도 확인하세요.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("[안내] 1~6: 모드 전환 / s: 저장 / q: 종료")

    mode_key = ord("1")
    fps_counter = FpsCounter(FPS_WINDOW)

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[오류] 프레임을 읽지 못했습니다. 카메라 연결이 끊겼는지 확인하세요.")
                break

            mode_name, process = MODES[mode_key]

            start = time.perf_counter()
            result = process(frame)
            process_ms = (time.perf_counter() - start) * 1000

            fps = fps_counter.tick()

            # 저장용 결과는 그대로 두고, 화면용 복사본에만 정보를 그린다.
            display = draw_overlay(result.copy(), mode_name, fps, process_ms)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key in MODES and key != mode_key:
                mode_key = key
                print(f"[모드] {MODES[mode_key][0]}")

                # HSV 모드를 벗어나면 트랙바 창을 닫는다.
                if key != ord("6") and is_window_open(HSV_WINDOW_NAME):
                    cv2.destroyWindow(HSV_WINDOW_NAME)

            if key == ord("s"):
                path = save_result(result, mode_name, OUTPUT_DIR)

                if path is not None:
                    print(f"[저장] {path}")
                else:
                    print("[오류] 결과 이미지를 저장하지 못했습니다.")

            # X 버튼으로 메인 창을 닫으면 종료한다.
            if not is_window_open(WINDOW_NAME):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[정보] 카메라와 창을 정리하고 종료했습니다.")


if __name__ == "__main__":
    main()


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 실행
#      python step5_opencv_practice.py
# 3. 창을 클릭한 뒤 숫자 키 1~6으로 모드를 바꾼다.
# 4. s 키를 누르면 outputs/ 폴더에 현재 처리 결과가 저장된다.
# 5. q 키(또는 창의 X 버튼)로 종료한다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 모드별 FPS와 처리 시간(proc ms) 비교 → 연산량이 실시간성에 미치는 영향
#    - 각 모드에서 proc ms 값을 표로 적어 본다. (원본 ≈ 0 ms → 블러/Canny는 수 ms)
#    - FPS는 카메라 최대 속도(약 30)에 막혀 있어 처리가 가벼우면 차이가 작다.
#    - 처리 시간이 한 프레임 간격(30FPS 기준 약 33 ms)을 넘으면 그때부터 FPS가 떨어진다.
#    - BLUR_KERNEL을 (51, 51), (99, 99)로 키워 proc ms와 FPS가 어떻게 바뀌는지 본다.
#    - STEP 6~7의 YOLO는 CPU에서 한 프레임에 수십~수백 ms가 걸린다 → FPS가 크게 떨어질 것을 예상해 본다.
# 2. Canny / 이진화
#    - 조명을 바꾸거나 손을 움직이며 결과가 어떻게 달라지는지 본다.
#    - CANNY_LOW/HIGH, THRESHOLD_VALUE를 바꿔 결과를 비교한다.
# 3. HSV 마스크
#    - 특정 색 물체(파란 펜, 빨간 컵 등)만 남도록 트랙바를 조절해 본다.
#    - 참고: 빨강은 H가 0 근처와 179 근처 양쪽에 걸쳐 있어 한 범위로 잡기 어렵다.
# 4. 그리기 함수
#    - draw_overlay의 cv2.rectangle, cv2.putText 좌표와 색 값을 바꿔 보며 사용법을 익힌다.
#    - 이 두 함수가 STEP 6~7에서 객체 박스와 라벨을 그리는 데 그대로 쓰인다.
