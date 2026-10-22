"""
1단계: 웹캠 열기와 실시간 미리보기.

OpenCV로 웹캠을 열고, 실제 적용된 해상도와 FPS를 확인한 뒤
화면에 실시간으로 보여준다.
q 키를 누르면 종료한다.
"""

import sys

import cv2

import config


def get_backend(use_dshow):
    """
    <summary>카메라를 열 때 사용할 OpenCV 백엔드 번호를 고른다.</summary>
    <param name="use_dshow">True이면 Windows에서 DirectShow 백엔드를 사용한다.</param>
    <input>config.USE_DSHOW 값</input>
    <output>cv2.CAP_DSHOW 또는 cv2.CAP_ANY (int)</output>
    <reason>
    Windows 기본 백엔드(MSMF)는 카메라가 켜지기까지 수 초 이상 걸리는 경우가 있다.
    DirectShow(CAP_DSHOW)는 대체로 더 빨리 열리지만 Windows 전용이므로
    다른 OS에서는 자동 선택(CAP_ANY)을 쓴다.
    </reason>
    """
    if use_dshow and sys.platform == "win32":
        return cv2.CAP_DSHOW

    return cv2.CAP_ANY


def find_available_cameras(max_index=4, backend=cv2.CAP_ANY):
    """
    <summary>0번부터 max_index-1번까지 카메라를 열어 보고 사용 가능한 번호를 찾는다.</summary>
    <param name="max_index">탐색할 인덱스 개수 (기본 4 → 0~3번)</param>
    <param name="backend">카메라를 열 때 사용할 OpenCV 백엔드</param>
    <input>없음 (실제 연결된 카메라 장치를 직접 조회)</input>
    <output>프레임을 실제로 읽을 수 있었던 카메라 번호 리스트 (list[int])</output>
    <reason>
    외장 웹캠, 가상 카메라(OBS 등)가 있으면 내장 카메라가 0번이 아닐 수 있다.
    isOpened()만으로는 부족한 경우가 있어 프레임을 한 장 읽어 보고 판단한다.
    </reason>
    """
    available = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index, backend)

        if cap.isOpened():
            ok, _ = cap.read()

            if ok:
                available.append(index)

        cap.release()

    return available


def open_camera(index, width, height, backend):
    """
    <summary>웹캠을 열고 원하는 해상도를 요청한다.</summary>
    <param name="index">카메라 번호</param>
    <param name="width">요청할 가로 해상도 (픽셀)</param>
    <param name="height">요청할 세로 해상도 (픽셀)</param>
    <param name="backend">카메라를 열 때 사용할 OpenCV 백엔드</param>
    <input>config의 CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT</input>
    <output>열린 cv2.VideoCapture 객체, 실패하면 None</output>
    <reason>
    set()은 "요청"일 뿐이라 카메라가 지원하지 않는 해상도는 무시될 수 있다.
    그래서 열기와 요청을 한 함수에 모으고, 실제 값은 호출한 쪽에서 get()으로 확인한다.
    </reason>
    """
    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def print_camera_info(cap):
    """
    <summary>카메라에 실제로 적용된 해상도와 FPS를 출력한다.</summary>
    <param name="cap">열린 cv2.VideoCapture 객체</param>
    <input>카메라 속성 (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS)</input>
    <output>없음 (콘솔 출력)</output>
    <reason>
    요청한 해상도와 실제 해상도가 다를 수 있음을 학생이 직접 확인하게 하기 위해서다.
    일부 드라이버는 FPS를 0으로 돌려주므로 그 경우를 따로 안내한다.
    </reason>
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"[정보] 요청 해상도: {config.FRAME_WIDTH} x {config.FRAME_HEIGHT}")
    print(f"[정보] 실제 해상도: {width} x {height}")

    if fps > 0:
        print(f"[정보] 카메라 FPS: {fps:.1f}")
    else:
        print("[정보] 카메라 FPS: 드라이버가 값을 알려주지 않음 (0)")


def print_frame_info(frame):
    """
    <summary>첫 프레임의 shape, dtype, 가운데 픽셀의 BGR 값을 출력한다.</summary>
    <param name="frame">카메라에서 읽은 이미지 (numpy 배열)</param>
    <input>cap.read()로 얻은 프레임 1장</input>
    <output>없음 (콘솔 출력)</output>
    <reason>
    OpenCV 이미지는 (높이, 너비, 채널) 순서의 uint8 배열이고
    채널 순서가 RGB가 아닌 BGR이라는 점이 이후 단계에서 자주 실수를 만든다.
    처음부터 숫자로 직접 보게 하려는 목적이다.
    </reason>
    """
    height, width = frame.shape[:2]
    blue, green, red = frame[height // 2, width // 2]

    print(f"[확인] frame.shape = {frame.shape}  → (높이, 너비, 채널)")
    print(f"[확인] frame.dtype = {frame.dtype}  → 0~255 정수")
    print(f"[확인] 가운데 픽셀 = B:{blue} G:{green} R:{red}  → 채널 순서는 BGR")


def main():
    """
    <summary>웹캠을 열어 실시간 미리보기를 보여주고 q 키로 종료한다.</summary>
    <input>config.py 설정값, 키보드 입력(q)</input>
    <output>미리보기 창, 콘솔 정보 출력</output>
    <reason>
    카메라 열기 → 정보 확인 → 반복 읽기/표시 → 자원 해제라는
    모든 실시간 영상 프로그램의 기본 뼈대를 한 함수에서 보여주기 위해서다.
    try/finally로 어떤 경우에도 카메라와 창이 정리되게 한다.
    </reason>
    """
    backend = get_backend(config.USE_DSHOW)

    print(f"[정보] 추론 장치 설정(DEVICE): {config.DEVICE}  (1단계에서는 사용하지 않음)")

    cap = open_camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, backend)

    if cap is None:
        print(f"[오류] {config.CAMERA_INDEX}번 카메라를 열 수 없습니다.")
        print("[안내] 사용 가능한 카메라를 찾는 중입니다...")

        available = find_available_cameras(backend=backend)

        if available:
            print(f"[안내] 사용 가능한 카메라 번호: {available}")
            print("[안내] config.py의 CAMERA_INDEX를 위 번호 중 하나로 바꿔 보세요.")
        else:
            print("[안내] 사용 가능한 카메라가 없습니다.")
            print("[안내] 카메라 연결, 다른 프로그램(Zoom, Teams 등) 사용 여부,")
            print("[안내] Windows 설정 > 개인 정보 > 카메라 권한을 확인하세요.")

        return

    print_camera_info(cap)
    print("[안내] 미리보기 창에서 q 키를 누르면 종료합니다.")

    first_frame = True

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[오류] 프레임을 읽지 못했습니다. 카메라 연결이 끊겼는지 확인하세요.")
                break

            if first_frame:
                print_frame_info(frame)
                first_frame = False

            cv2.imshow("Step1 - Webcam (q: quit)", frame)

            # waitKey가 있어야 창이 갱신된다. 1ms 동안 키 입력을 기다린다.
            if cv2.waitKey(1) & 0xFF == ord("q"):
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
#      python step1_webcam.py
# 3. 미리보기 창을 클릭한 뒤 q 키를 누르면 종료된다.
# 4. 카메라가 켜지기까지 오래 걸리면 config.py에서 USE_DSHOW = True 로 바꾼다.
# 5. 카메라를 못 찾으면 출력된 "사용 가능한 카메라 번호"로 CAMERA_INDEX를 바꾼다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. frame.shape
#    - (480, 640, 3)처럼 (높이, 너비, 채널) 순서다.
#    - 해상도는 "640 x 480"(너비 x 높이)로 말하지만 배열은 높이가 먼저다.
# 2. frame.dtype
#    - uint8: 각 픽셀 채널 값은 0~255 정수다.
# 3. BGR 채널 순서
#    - OpenCV는 RGB가 아니라 BGR 순서로 저장한다.
#    - 빨간 물체를 화면 가운데에 대면 세 번째 값(R)이 커지는지 확인해 본다.
# 4. 요청 해상도 vs 실제 해상도
#    - FRAME_WIDTH/HEIGHT를 1920x1080 등으로 바꿔 실제 값이 어떻게 되는지 비교한다.
# 5. USE_DSHOW를 True/False로 바꿔 카메라가 켜지는 시간을 비교한다.
