"""
실습 공통 설정 파일.

모든 단계(step1 ~ step7)가 이 파일을 import 해서 같은 설정을 사용한다.
"""

# 사용할 웹캠 번호 (노트북 내장 카메라는 보통 0번)
CAMERA_INDEX = 0

# 요청할 해상도 (카메라가 지원하지 않으면 가장 가까운 값으로 바뀐다)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Windows에서 카메라가 늦게 켜지면 True로 바꿔 DirectShow 백엔드를 사용한다
USE_DSHOW = False

# YOLO 모델 파일 (없으면 ultralytics가 처음 실행할 때 자동으로 내려받는다)
MODEL_PATH = "yolo26n.pt"

# MODEL_PATH를 불러오지 못할 때 대신 시도할 모델 (구버전 ultralytics에도 있는 모델)
FALLBACK_MODEL_PATH = "yolo11n.pt"

# YOLO 입력 크기 (32의 배수, 작을수록 빠르고 작은 물체를 놓치기 쉽다)
IMG_SIZE = 640

# 이 신뢰도보다 낮은 탐지 결과는 버린다
CONF_THRESHOLD = 0.4

# 추론 장치 설정
# "cpu"    : GPU 없는 노트북 (기본값)
# "cuda:0" : NVIDIA GPU
# "mps"    : Apple Silicon Mac
DEVICE = "cpu"
