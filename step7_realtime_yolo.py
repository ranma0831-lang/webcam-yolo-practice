"""
7단계: 웹캠 실시간 YOLO 객체 인식.

두 가지 실행 모드를 지원한다.
  --mode window : cv2.imshow 창으로 보기 (q: 종료)
  --mode web    : 3단계 Flask 서버에 탐지 결과 스트림(/detect_feed)을 더해 브라우저로 보기

앞 단계의 코드를 재사용한다.
  step2: Camera, generate_frames  /  step3: 캡처 파일명·이력  /  step5: FpsCounter
  step6: load_model, draw_detections
"""

import argparse
import atexit
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, send_from_directory
from ultralytics import YOLO

import config
from step2_server import HOST, PORT, JPEG_QUALITY, Camera, generate_frames, get_backend
from step3_capture import HISTORY_LIMIT, list_recent_captures, make_capture_path
from step5_opencv_practice import FpsCounter
from step6_yolo_image import draw_detections, load_model


# 탐지 결과를 촬영해 저장할 폴더 (3단계와 같은 폴더)
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

WINDOW_NAME = "Step7 - Realtime YOLO (q: quit)"

# 장치별 권장 기본값 (명령줄에서 직접 지정하면 그 값을 우선한다)
CPU_DEFAULTS = {"imgsz": 320, "skip": 2}
GPU_DEFAULTS = {"imgsz": 640, "skip": 1}

# FPS 이동평균에 사용할 최근 프레임 수
FPS_WINDOW = 30

# web 모드에서 추론하지 않는 프레임의 루프 간격 (카메라 약 30FPS에 맞춤)
IDLE_INTERVAL = 1 / 30

# 화면 표시 색 (BGR, 채도 낮은 톤)
PANEL_COLOR = (40, 40, 40)
TEXT_COLOR = (235, 235, 235)
ACCENT_COLOR = (150, 200, 120)


def parse_args():
    """
    <summary>명령줄 옵션을 읽는다.</summary>
    <input>python step7_realtime_yolo.py --mode web --imgsz 416 ... 형태의 인자</input>
    <output>argparse.Namespace (옵션 값 모음)</output>
    <reason>
    imgsz, skip, 장치 등을 코드 수정 없이 바꿔 가며 FPS를 비교하는 실험을 하기 위해서다.
    imgsz와 skip의 기본값을 None으로 두어, 사용자가 지정하지 않았을 때만 장치별 권장값을 채운다.
    </reason>
    """
    parser = argparse.ArgumentParser(description="웹캠 실시간 YOLO 객체 인식")

    parser.add_argument("--mode", choices=["window", "web"], default="window", help="window: OpenCV 창, web: Flask 브라우저")
    parser.add_argument("--device", choices=["cpu", "cuda:0", "mps"], default=config.DEVICE, help="추론 장치")
    parser.add_argument("--imgsz", type=int, default=None, help="입력 크기 (기본: CPU 320 / GPU 640)")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD, help="신뢰도 기준값 (0~1)")
    parser.add_argument("--skip", type=int, default=None, help="N프레임마다 1번 추론 (기본: CPU 2 / GPU 1)")
    parser.add_argument("--model", default=config.MODEL_PATH, help="모델 파일 (예: yolo26n.pt)")
    parser.add_argument("--openvino", action="store_true", help="OpenVINO 변환 모델로 실행 (Intel CPU 가속)")
    parser.add_argument("--classes", nargs="+", default=None, help="이 클래스만 탐지 (예: --classes person cup)")

    return parser.parse_args()


def check_device(device):
    """
    <summary>요청한 장치를 이 컴퓨터에서 쓸 수 있는지 확인하고, 안 되면 cpu로 바꾼다.</summary>
    <param name="device">요청한 장치 ("cpu", "cuda:0", "mps")</param>
    <input>--device 옵션 값</input>
    <output>실제로 사용할 장치 문자열</output>
    <reason>
    GPU가 없는 노트북에서 --device cuda:0을 주면 추론 중에 알아보기 어려운 오류가 난다.
    시작할 때 미리 확인해 원인을 알려주고 CPU로 계속 실습할 수 있게 한다.
    </reason>
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[경고] CUDA GPU를 찾을 수 없습니다. (NVIDIA GPU와 CUDA용 PyTorch가 필요)")
        print("[경고] cpu로 바꿔서 실행합니다.")
        return "cpu"

    if device == "mps" and not torch.backends.mps.is_available():
        print("[경고] Apple MPS를 사용할 수 없습니다. (Apple Silicon Mac이 필요)")
        print("[경고] cpu로 바꿔서 실행합니다.")
        return "cpu"

    return device


def apply_device_defaults(args):
    """
    <summary>사용자가 지정하지 않은 imgsz, skip을 장치에 맞는 권장값으로 채운다.</summary>
    <param name="args">parse_args() 결과 (직접 수정됨)</param>
    <input>--device, --imgsz, --skip 옵션 값</input>
    <output>같은 args (imgsz, skip, half 값이 채워짐)</output>
    <reason>
    CPU는 추론이 느려 작은 입력(320)과 프레임 건너뛰기(skip=2)로 실시간성을 확보한다.
    GPU는 충분히 빨라 큰 입력(640)으로 매 프레임(skip=1) 추론해도 된다.
    half(16비트 연산)는 CUDA GPU에서만 속도 이득이 있다.
    </reason>
    """
    is_gpu = args.device != "cpu"
    defaults = GPU_DEFAULTS if is_gpu else CPU_DEFAULTS

    if args.imgsz is None:
        args.imgsz = defaults["imgsz"]

    if args.skip is None:
        args.skip = defaults["skip"]

    args.skip = max(1, args.skip)
    args.half = args.device.startswith("cuda")

    return args


def load_detector_model(args):
    """
    <summary>옵션에 따라 OpenVINO 모델 또는 PyTorch(.pt) 모델을 불러온다.</summary>
    <param name="args">옵션 값 (model, openvino, device 사용)</param>
    <input>모델 파일 또는 OpenVINO 변환 폴더</input>
    <output>(YOLO 모델, 백엔드 이름 "openvino" 또는 "pytorch"), 실패하면 (None, None)</output>
    <reason>
    OpenVINO는 Intel이 만든 추론 엔진으로, Intel CPU에서 PyTorch보다 빠르게 동작한다.
    변환 폴더가 없으면 만드는 방법을 안내하고 .pt 모델로 계속 실행한다.
    GPU(cuda/mps)를 쓸 때는 OpenVINO가 의미 없으므로 무시한다.
    </reason>
    """
    if args.openvino:
        model_path = Path(args.model)
        openvino_dir = model_path.parent / f"{model_path.stem}_openvino_model"

        if args.device != "cpu":
            print("[안내] --openvino는 CPU 전용 옵션이라 GPU 실행에서는 무시합니다.")
        elif openvino_dir.is_dir():
            print(f"[정보] OpenVINO 모델을 불러옵니다: {openvino_dir}")

            try:
                return YOLO(str(openvino_dir), task="detect"), "openvino"
            except Exception as error:
                print(f"[오류] OpenVINO 모델을 불러오지 못했습니다: {error}")
                print("[안내] pip install openvino 로 설치되어 있는지 확인하세요.")
        else:
            print(f"[안내] '{openvino_dir}' 폴더가 없습니다. 아래 명령으로 먼저 변환하세요.")
            print("[안내]   pip install openvino")
            print(f"[안내]   yolo export model={args.model} format=openvino")

        print("[안내] PyTorch(.pt) 모델로 실행합니다.")

    model = load_model(args.model, config.FALLBACK_MODEL_PATH)

    if model is None:
        return None, None

    return model, "pytorch"


def resolve_class_ids(class_names, names):
    """
    <summary>--classes로 받은 클래스 이름을 모델의 클래스 번호로 바꾼다.</summary>
    <param name="class_names">사용자가 입력한 클래스 이름 리스트 (예: ["person", "cup"]), 없으면 None</param>
    <param name="names">모델의 클래스 번호 → 이름 딕셔너리 (model.names)</param>
    <input>--classes 옵션 값</input>
    <output>클래스 번호 리스트, 필터가 없으면 None. 모르는 이름이 있으면 ValueError</output>
    <reason>
    predict의 classes 옵션은 번호만 받는다.
    학생이 이름으로 입력하고, 오타가 있으면 사용 가능한 이름을 보여줘 바로 고치게 한다.
    </reason>
    """
    if not class_names:
        return None

    name_to_id = {name: class_id for class_id, name in names.items()}
    unknown = [name for name in class_names if name not in name_to_id]

    if unknown:
        examples = ", ".join(list(name_to_id)[:20])
        raise ValueError(f"모르는 클래스 이름: {unknown}\n[안내] 사용 가능한 이름 예: {examples} ...")

    return [name_to_id[name] for name in class_names]


class Detector:
    """
    <summary>N프레임마다 YOLO 추론을 하고, 나머지 프레임에는 직전 결과를 재사용하는 클래스.</summary>
    <param name="model">YOLO 모델 객체</param>
    <param name="args">옵션 값 (device, imgsz, conf, skip, half 사용)</param>
    <param name="class_ids">탐지할 클래스 번호 리스트 (None이면 전체)</param>
    <input>process()에 매 프레임 BGR 이미지를 넣는다.</input>
    <output>process()가 (최근 탐지 결과, 이번 프레임에 추론했는지)를 돌려준다.</output>
    <reason>
    CPU 추론은 한 번에 수십 ms가 걸려 매 프레임 추론하면 FPS가 크게 떨어진다.
    사람이나 물체는 1/30초 사이에 거의 움직이지 않으므로
    직전 박스를 그대로 다시 그려도 눈으로는 차이가 작고, 화면은 더 부드러워진다.
    </reason>
    """

    def __init__(self, model, args, class_ids):
        """
        <summary>모델과 추론 옵션을 저장한다.</summary>
        <param name="model">YOLO 모델 객체</param>
        <param name="args">옵션 값</param>
        <param name="class_ids">탐지할 클래스 번호 리스트 (None이면 전체)</param>
        <input>없음</input>
        <output>없음</output>
        <reason>predict 옵션을 한 곳에 모아 매 프레임 같은 조건으로 추론하게 한다.</reason>
        """
        self.model = model
        self.names = model.names
        self.frame_index = 0
        self.skip = args.skip
        self.last_result = None
        self.last_infer_ms = 0.0

        self.options = {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "device": args.device,
            "classes": class_ids,
            "verbose": False,
        }

        # half는 CUDA에서만 넘긴다. (ultralytics 8.4부터 폐지 예정 경고가 있어 CPU에서는 생략)
        if args.half:
            self.options["half"] = True

    def process(self, frame):
        """
        <summary>skip 간격에 맞춰 추론하거나 직전 결과를 돌려준다.</summary>
        <param name="frame">BGR 프레임 (numpy 배열)</param>
        <input>웹캠 프레임</input>
        <output>(results[0] 객체, 이번 프레임에 추론했으면 True)</output>
        <reason>
        frame_index를 skip으로 나눈 나머지가 0일 때만 추론한다.
        첫 프레임은 결과가 없으므로 반드시 추론한다.
        </reason>
        """
        should_infer = self.last_result is None or self.frame_index % self.skip == 0
        self.frame_index += 1

        if not should_infer:
            return self.last_result, False

        start = time.perf_counter()
        results = self.model.predict(source=frame, **self.options)
        self.last_infer_ms = (time.perf_counter() - start) * 1000
        self.last_result = results[0]

        return self.last_result, True


def count_classes(result, names):
    """
    <summary>탐지 결과를 클래스별 개수로 집계한다.</summary>
    <param name="result">results[0] 객체</param>
    <param name="names">클래스 번호 → 이름 딕셔너리</param>
    <input>result.boxes.cls</input>
    <output>Counter (예: {"person": 2, "cup": 1})</output>
    <reason>"사람 몇 명, 컵 몇 개"처럼 숫자로 바꾸면 출입 인원 세기 같은 응용으로 이어진다.</reason>
    """
    classes = result.boxes.cls.cpu().numpy().astype(int)

    return Counter(names[c] for c in classes)


def draw_status(image, fps, infer_ms, counts, device_label):
    """
    <summary>화면 위쪽에 FPS·추론 시간·탐지 개수·장치를, 아래쪽에 클래스별 개수를 그린다.</summary>
    <param name="image">박스가 그려진 BGR 이미지 (직접 수정됨)</param>
    <param name="fps">이동평균 FPS</param>
    <param name="infer_ms">가장 최근 추론에 걸린 시간 (ms)</param>
    <param name="counts">클래스별 개수 Counter</param>
    <param name="device_label">장치와 설정 요약 문자열 (영문)</param>
    <input>draw_detections 결과 이미지</input>
    <output>정보가 그려진 같은 이미지</output>
    <reason>
    STEP 5의 rectangle + putText 방식 그대로, 실험에 필요한 숫자를 화면에서 바로 읽게 한다.
    FPS와 추론 ms를 함께 보면 "추론은 느려도 skip 덕분에 화면은 부드럽다"는 것을 확인할 수 있다.
    </reason>
    """
    height, width = image.shape[:2]
    total = sum(counts.values())

    cv2.rectangle(image, (0, 0), (width, 52), PANEL_COLOR, thickness=-1)

    line1 = f"FPS {fps:5.1f} | infer {infer_ms:6.1f} ms | objects {total}"
    cv2.putText(image, line1, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(image, device_label, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    if counts:
        summary = "  ".join(f"{name}:{count}" for name, count in counts.most_common())
    else:
        summary = "no objects"

    cv2.rectangle(image, (0, height - 28), (width, height), PANEL_COLOR, thickness=-1)
    cv2.putText(image, summary, (10, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)

    return image


def annotate_frame(frame, detector, fps_counter, device_label):
    """
    <summary>프레임 1장에 대해 추론(또는 재사용) → 박스 그리기 → 상태 표시를 한 번에 처리한다.</summary>
    <param name="frame">BGR 프레임</param>
    <param name="detector">Detector 객체</param>
    <param name="fps_counter">FpsCounter 객체</param>
    <param name="device_label">화면에 표시할 장치·설정 요약</param>
    <input>웹캠 프레임</input>
    <output>(정보가 그려진 이미지, 이번 프레임에 추론했는지)</output>
    <reason>window 모드와 web 모드가 같은 처리 과정을 쓰도록 한 함수로 묶는다.</reason>
    """
    result, inferred = detector.process(frame)
    fps = fps_counter.tick()

    canvas = draw_detections(frame, result, detector.names)
    counts = count_classes(result, detector.names)
    draw_status(canvas, fps, detector.last_infer_ms, counts, device_label)

    return canvas, inferred


def run_window_mode(detector, device_label):
    """
    <summary>웹캠을 열고 탐지 결과를 OpenCV 창으로 보여준다.</summary>
    <param name="detector">Detector 객체</param>
    <param name="device_label">화면에 표시할 장치·설정 요약</param>
    <input>웹캠 프레임, 키보드 입력(q)</input>
    <output>탐지 결과 창</output>
    <reason>
    STEP 5의 "읽기 → 처리 → 그리기 → 표시" 루프에서 처리 자리에 YOLO를 넣은 형태다.
    서버 없이 가장 단순하게 실시간 탐지를 확인할 수 있다.
    </reason>
    """
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

    fps_counter = FpsCounter(FPS_WINDOW)

    print("[안내] 창에서 q 키를 누르면(또는 창을 닫으면) 종료합니다.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("[오류] 프레임을 읽지 못했습니다. 카메라 연결이 끊겼는지 확인하세요.")
                break

            canvas, _ = annotate_frame(frame, detector, fps_counter, device_label)
            cv2.imshow(WINDOW_NAME, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[정보] 카메라와 창을 정리하고 종료했습니다.")


class DetectionStream:
    """
    <summary>Camera의 최신 프레임에 탐지 결과를 그려, 최신 결과 프레임 1장을 공유하는 클래스.</summary>
    <param name="camera">STEP 2의 Camera 객체 (원본 프레임 제공)</param>
    <param name="detector">Detector 객체</param>
    <param name="device_label">화면에 표시할 장치·설정 요약</param>
    <input>camera.get_frame()으로 얻은 원본 프레임</input>
    <output>get_frame()으로 탐지 결과가 그려진 최신 프레임을 돌려준다.</output>
    <reason>
    추론을 브라우저 요청마다 하면 탭을 여러 개 열 때 같은 프레임을 여러 번 추론하게 된다.
    STEP 2의 Camera와 같은 구조(별도 스레드 + 최신 1장 + lock)로 추론을 한 곳에서만 하고,
    get_frame()과 running을 Camera와 똑같이 맞춰 generate_frames를 그대로 재사용한다.
    </reason>
    """

    def __init__(self, camera, detector, device_label):
        """
        <summary>추론 스레드를 시작한다.</summary>
        <param name="camera">Camera 객체</param>
        <param name="detector">Detector 객체</param>
        <param name="device_label">화면에 표시할 장치·설정 요약</param>
        <input>없음</input>
        <output>없음</output>
        <reason>추론은 오래 걸리므로 요청 처리 스레드와 분리한다.</reason>
        """
        self.camera = camera
        self.detector = detector
        self.device_label = device_label
        self.fps_counter = FpsCounter(FPS_WINDOW)

        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    @property
    def running(self):
        """
        <summary>카메라와 추론 스레드가 모두 동작 중인지 알려준다.</summary>
        <input>없음</input>
        <output>bool</output>
        <reason>generate_frames가 Camera.running과 같은 방식으로 스트림 종료를 판단하게 한다.</reason>
        """
        return self.camera.running and not self.stopped

    def _update(self):
        """
        <summary>추론 스레드 본체. 최신 원본 프레임에 탐지 결과를 그려 self.frame을 교체한다.</summary>
        <input>camera.get_frame()</input>
        <output>없음 (self.frame 갱신)</output>
        <reason>
        추론하지 않는 프레임은 처리가 매우 빨라 같은 프레임을 계속 반복하게 되므로
        카메라 속도에 맞춰 잠깐 쉰다.
        추론 중 오류가 나면 원인을 출력하고 스트림을 멈춰 브라우저가 멈춘 화면을 계속 받지 않게 한다.
        </reason>
        """
        while self.running:
            frame = self.camera.get_frame()

            if frame is None:
                time.sleep(0.01)
                continue

            try:
                canvas, inferred = annotate_frame(frame, self.detector, self.fps_counter, self.device_label)
            except Exception as error:
                print(f"[오류] 추론 중 문제가 발생했습니다: {error}")
                self.stopped = True
                break

            with self.lock:
                self.frame = canvas

            if not inferred:
                time.sleep(IDLE_INTERVAL)

    def get_frame(self):
        """
        <summary>탐지 결과가 그려진 최신 프레임의 복사본을 돌려준다.</summary>
        <input>추론 스레드가 저장한 self.frame</input>
        <output>numpy 배열 (아직 없으면 None)</output>
        <reason>Camera.get_frame과 같은 이유로 lock 안에서 복사본을 만든다.</reason>
        """
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        """
        <summary>추론 스레드를 멈춘다.</summary>
        <input>없음</input>
        <output>없음</output>
        <reason>종료 시 추론 스레드가 해제된 카메라에 계속 접근하지 않도록 먼저 멈춘다.</reason>
        """
        self.stopped = True

        if self.thread.is_alive():
            self.thread.join(timeout=2)


def create_web_app(camera, stream, capture_dir):
    """
    <summary>STEP 3 서버 구조에 탐지 결과 스트림(/detect_feed)을 더한 Flask 앱을 만든다.</summary>
    <param name="camera">원본 프레임을 주는 Camera 객체 (/video_feed)</param>
    <param name="stream">탐지 결과 프레임을 주는 DetectionStream 객체 (/detect_feed, /capture)</param>
    <param name="capture_dir">촬영 이미지 저장 폴더 (Path)</param>
    <input>Camera, DetectionStream, 저장 폴더</input>
    <output>설정이 끝난 Flask 앱</output>
    <reason>
    STEP 3와 같은 페이지(capture.html)를 쓰되, 스트림 주소만 /detect_feed로 바꿔 넘긴다.
    촬영 버튼은 원본이 아니라 탐지 결과 프레임을 저장해 박스가 그려진 장면을 기록으로 남긴다.
    원본 스트림(/video_feed)은 비교용으로 그대로 둔다.
    </reason>
    """
    app = Flask(__name__)

    @app.route("/")
    def index():
        """
        <summary>탐지 스트림, 촬영 버튼, 촬영 이력을 보여주는 페이지를 돌려준다.</summary>
        <input>브라우저의 GET / 요청</input>
        <output>templates/capture.html을 탐지용 문구로 렌더링한 HTML</output>
        <reason>STEP 3 템플릿을 재사용해 화면 구성을 학생에게 익숙하게 유지한다.</reason>
        """
        return render_template(
            "capture.html",
            page_title="실시간 YOLO 탐지",
            subtitle="촬영 버튼은 박스가 그려진 탐지 결과 프레임을 저장합니다.",
            feed_url="/detect_feed",
            feed_caption="탐지 결과 (/detect_feed)",
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            captures=list_recent_captures(capture_dir, HISTORY_LIMIT),
            history_limit=HISTORY_LIMIT,
        )

    @app.route("/video_feed")
    def video_feed():
        """
        <summary>원본 웹캠 MJPEG 스트림 (STEP 2와 동일).</summary>
        <input>GET /video_feed</input>
        <output>multipart/x-mixed-replace 응답</output>
        <reason>탐지 결과와 원본을 브라우저 탭 두 개로 비교해 볼 수 있게 남겨 둔다.</reason>
        """
        return Response(generate_frames(camera), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/detect_feed")
    def detect_feed():
        """
        <summary>탐지 결과가 그려진 MJPEG 스트림.</summary>
        <input>GET /detect_feed</input>
        <output>multipart/x-mixed-replace 응답</output>
        <reason>DetectionStream이 Camera와 같은 모양이라 generate_frames를 그대로 쓴다.</reason>
        """
        return Response(generate_frames(stream), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/capture", methods=["POST"])
    def capture():
        """
        <summary>현재 탐지 결과 프레임을 jpg로 저장하고 파일명을 JSON으로 돌려준다.</summary>
        <input>POST /capture</input>
        <output>성공: {"filename": "..."} (201), 실패: {"error": "..."} (503 또는 500)</output>
        <reason>
        STEP 3와 같은 파일명 규칙(make_capture_path)을 쓰고,
        한글 경로 문제를 피하려고 imencode + write_bytes로 저장한다.
        </reason>
        """
        frame = stream.get_frame()

        if frame is None:
            return jsonify(error="아직 탐지 결과 프레임이 없습니다. 잠시 후 다시 시도하세요."), 503

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if not ok:
            return jsonify(error="이미지를 JPEG로 변환하지 못했습니다."), 500

        path = make_capture_path(capture_dir)

        try:
            path.write_bytes(jpeg.tobytes())
        except OSError as error:
            print(f"[오류] 캡처 저장 실패: {error}")
            return jsonify(error="이미지를 저장하지 못했습니다. 폴더 권한을 확인하세요."), 500

        print(f"[정보] 탐지 결과 저장: {path}")

        return jsonify(filename=path.name), 201

    @app.route("/captures/<path:filename>")
    def captured_image(filename):
        """
        <summary>captures/ 폴더 안의 저장 이미지를 돌려준다.</summary>
        <param name="filename">요청한 파일 이름</param>
        <input>GET /captures/파일명</input>
        <output>jpg 이미지, 폴더 밖 경로이거나 없는 파일이면 404</output>
        <reason>send_from_directory가 ../ 같은 경로 조작을 막아 준다 (STEP 3와 동일).</reason>
        """
        return send_from_directory(capture_dir, filename)

    return app


def run_web_mode(detector, device_label):
    """
    <summary>카메라와 추론 스레드를 시작하고 Flask 서버를 실행한다.</summary>
    <param name="detector">Detector 객체</param>
    <param name="device_label">화면에 표시할 장치·설정 요약</param>
    <input>config.py 카메라 설정</input>
    <output>http://127.0.0.1:5000 웹 서버</output>
    <reason>
    카메라 읽기 스레드(STEP 2) → 추론 스레드(DetectionStream) → 요청 스레드(Flask)로
    역할을 나눠, 느린 추론이 카메라 읽기나 페이지 응답을 막지 않게 한다.
    종료할 때는 추론 스레드를 먼저 멈추고 카메라를 해제한다.
    </reason>
    """
    CAPTURE_DIR.mkdir(exist_ok=True)

    backend = get_backend(config.USE_DSHOW)

    try:
        camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, backend)
    except RuntimeError as error:
        print(f"[오류] {error}")
        print("[안내] step1_webcam.py로 사용 가능한 카메라 번호를 확인하고 config.py의 CAMERA_INDEX를 바꿔 보세요.")
        print("[안내] 다른 프로그램(step2/step3 서버, Zoom 등)이 카메라를 쓰고 있지 않은지도 확인하세요.")
        return

    stream = DetectionStream(camera, detector, device_label)

    # atexit은 나중에 등록한 것부터 실행한다 → stream.stop() 후 camera.release()
    atexit.register(camera.release)
    atexit.register(stream.stop)

    app = create_web_app(camera, stream, CAPTURE_DIR)

    print(f"[안내] 브라우저에서 http://{HOST}:{PORT} 에 접속하세요.")
    print(f"[안내] 원본 스트림: http://{HOST}:{PORT}/video_feed")
    print("[안내] 종료하려면 이 터미널에서 Ctrl+C를 누르세요.")

    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)


def main():
    """
    <summary>옵션을 정리하고 모델을 불러온 뒤 선택한 모드(window/web)로 실행한다.</summary>
    <input>명령줄 옵션, config.py 설정값</input>
    <output>탐지 결과 창 또는 웹 서버</output>
    <reason>
    장치 확인 → 권장값 적용 → 모델 로드 → 클래스 필터 → 워밍업 순서로 준비를 끝낸 뒤 실행해,
    문제가 생기면 실시간 화면이 뜨기 전에 원인을 알려준다.
    </reason>
    """
    args = parse_args()
    args.device = check_device(args.device)
    apply_device_defaults(args)

    model, backend_name = load_detector_model(args)

    if model is None:
        print("[오류] 사용할 수 있는 YOLO 모델이 없어 종료합니다.")
        return

    try:
        class_ids = resolve_class_ids(args.classes, model.names)
    except ValueError as error:
        print(f"[오류] {error}")
        return

    detector = Detector(model, args, class_ids)

    device_label = f"{args.device} ({backend_name}) | imgsz {args.imgsz} | skip {args.skip} | conf {args.conf}"

    if args.half:
        device_label += " | half"

    print(f"[설정] mode={args.mode}, {device_label}")

    if class_ids is not None:
        print(f"[설정] 클래스 필터: {args.classes}")

    # 워밍업: 첫 추론은 모델 준비 시간이 섞여 수 초가 걸릴 수 있다.
    print("[정보] 모델 워밍업 중... (처음 한 번은 몇 초 걸릴 수 있습니다)")

    try:
        warmup_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        detector.model.predict(source=warmup_frame, **detector.options)
    except Exception as error:
        print(f"[오류] 워밍업 추론에 실패했습니다: {error}")
        print("[안내] --device 설정이 이 컴퓨터에 맞는지 확인하세요.")
        return

    if args.mode == "window":
        run_window_mode(detector, device_label)
    else:
        run_web_mode(detector, device_label)


if __name__ == "__main__":
    main()


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. 창 모드 (CPU 권장값 imgsz=320, skip=2 자동 적용)
#      python step7_realtime_yolo.py
# 3. 웹 모드 → 브라우저에서 http://127.0.0.1:5000 접속
#      python step7_realtime_yolo.py --mode web
# 4. 옵션 예시
#      python step7_realtime_yolo.py --imgsz 640 --skip 1
#      python step7_realtime_yolo.py --classes person cup "cell phone"   (이름에 공백이 있으면 따옴표로 감싼다)
#      python step7_realtime_yolo.py --device cuda:0                   (NVIDIA GPU: imgsz 640, skip 1, half 자동)
# 5. OpenVINO (Intel CPU 가속, 선택)
#      pip install openvino
#      yolo export model=yolo26n.pt format=openvino imgsz=320
#      python step7_realtime_yolo.py --openvino
# 6. step2/step3 서버나 다른 카메라 프로그램이 켜져 있으면 먼저 끈다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 아래 표를 공책에 그리고, 설정을 바꿔 실행하며 화면의 FPS와 infer ms를 기록한다.
#    (같은 장면에서 10초 정도 지켜본 뒤 안정된 값을 적는다)
#
#    | 장치            | imgsz | skip | FPS | infer ms | 탐지 개수 | 느낀 점 (끊김, 박스 지연 등) |
#    |-----------------|-------|------|-----|----------|-----------|------------------------------|
#    | cpu             | 320   | 1    |     |          |           |                              |
#    | cpu             | 320   | 2    |     |          |           |                              |
#    | cpu             | 320   | 3    |     |          |           |                              |
#    | cpu             | 640   | 1    |     |          |           |                              |
#    | cpu             | 640   | 2    |     |          |           |                              |
#    | cpu (openvino)  | 320   | 1    |     |          |           |                              |
#    | cuda:0 / mps    | 640   | 1    |     |          |           |                              |
#
# 2. CPU vs GPU
#    - GPU는 infer ms가 크게 줄어 skip=1, imgsz=640에서도 카메라 최대 FPS(약 30)에 가까워진다.
#    - GPU가 없으면 이 줄은 비워 두고, 다른 학생이나 강사 시연 값을 적는다.
# 3. imgsz
#    - 320 → 640으로 키우면 infer ms가 약 3~4배 늘어난다 (픽셀 수가 4배).
#    - 대신 멀리 있는 작은 물체(컵, 휴대폰)가 더 잘 잡히는지 비교한다.
# 4. skip
#    - skip을 키우면 FPS는 오르지만, 빠르게 움직일 때 박스가 물체를 늦게 따라온다.
#    - "화면이 부드러운 것"과 "박스가 정확한 것"은 서로 다른 지표임을 확인한다.
# 5. 클래스 필터와 개수 집계
#    - --classes person 으로 사람만 남기고, 아래쪽 집계(person:N)가 사람 수와 맞는지 본다.
