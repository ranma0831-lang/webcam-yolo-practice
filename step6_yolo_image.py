"""
6단계: 캡처 이미지 1장에 YOLO 객체 인식 적용하기.

3단계에서 저장한 captures/ 폴더의 가장 최근 jpg를 입력으로 쓴다.
캡처 이미지가 없으면 웹캠에서 1장을 찍어 대신 사용한다.
탐지 결과를 표로 출력하고, 직접 그린 결과와 results[0].plot() 결과를 나란히 비교한다.
"""

import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config


# 3단계 캡처 이미지 폴더와 결과 저장 폴더
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# 웹캠 폴백 시 버리는 프레임 수 (켜자마자 찍으면 노출이 맞지 않아 어둡다)
WARMUP_FRAMES = 10

# 입력 크기별 속도 비교에 사용할 크기와 반복 횟수
COMPARE_SIZES = (320, 416, 640)
COMPARE_REPEATS = 5

# 박스 색 (BGR, 채도 낮은 톤). 클래스 번호로 골라 같은 클래스는 같은 색이 되게 한다.
BOX_COLORS = [
    (150, 200, 120),
    (200, 160, 110),
    (120, 160, 210),
    (190, 130, 170),
    (110, 190, 200),
    (170, 170, 120),
]
LABEL_TEXT_COLOR = (30, 30, 30)


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
    Windows에서는 경로에 한글이 섞이면 cv2.imread가 None을 돌려준다.
    numpy로 파일 바이트를 읽고 cv2.imdecode로 해석하면 경로와 상관없이 읽을 수 있다.
    </reason>
    """
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


def load_model(model_path, fallback_path):
    """
    <summary>YOLO 모델을 불러오고, 실패하면 대체 모델을 안내하고 시도한다.</summary>
    <param name="model_path">우선 불러올 모델 파일 이름 (예: yolo26n.pt)</param>
    <param name="fallback_path">실패 시 시도할 모델 파일 이름 (예: yolo11n.pt)</param>
    <input>모델 파일 (없으면 ultralytics가 인터넷에서 자동으로 내려받음)</input>
    <output>YOLO 모델 객체, 둘 다 실패하면 None</output>
    <reason>
    yolo26n.pt는 최신 ultralytics에서만 지원하고, 처음 실행 시 내려받기가 필요하다.
    설치된 버전이 낮거나 인터넷이 막혀 있으면 실패하므로 원인과 대안을 친절하게 알려준다.
    </reason>
    """
    try:
        return YOLO(model_path)
    except Exception as error:
        print(f"[오류] 모델 '{model_path}'을(를) 불러오지 못했습니다: {error}")
        print("[안내] ultralytics 버전이 낮거나 인터넷 연결 문제로 내려받지 못했을 수 있습니다.")
        print("[안내] pip install -U ultralytics 로 업데이트하거나,")
        print(f"[안내] config.py의 MODEL_PATH를 \"{fallback_path}\"로 바꿔 보세요.")
        print(f"[안내] 지금은 '{fallback_path}'로 다시 시도합니다.")

    try:
        return YOLO(fallback_path)
    except Exception as error:
        print(f"[오류] 대체 모델 '{fallback_path}'도 불러오지 못했습니다: {error}")
        return None


def run_predict(model, image, imgsz):
    """
    <summary>YOLO 추론을 1번 실행하고 걸린 시간(ms)을 함께 돌려준다.</summary>
    <param name="model">YOLO 모델 객체</param>
    <param name="image">BGR 이미지 (numpy 배열)</param>
    <param name="imgsz">모델 입력 크기 (예: 320, 416, 640)</param>
    <input>cv2로 읽은 BGR 이미지 (ultralytics는 numpy 입력을 BGR로 가정한다)</input>
    <output>(results 리스트, 전체 소요 시간 ms)</output>
    <reason>
    predict 옵션을 한 곳에 모아 모든 측정이 같은 조건으로 실행되게 한다.
    DEVICE가 cuda일 때만 half=True(16비트 연산)를 써서 GPU 속도를 높인다.
    CPU는 16비트 연산 이득이 없어 옵션 자체를 넘기지 않는다.
    (ultralytics 8.4부터 half는 폐지 예정 경고가 나오므로 CPU에서 불필요한 경고를 피한다.)
    </reason>
    """
    options = {}

    if config.DEVICE.startswith("cuda"):
        options["half"] = True

    start = time.perf_counter()
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=config.CONF_THRESHOLD,
        device=config.DEVICE,
        verbose=False,
        **options,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    return results, elapsed_ms


def print_detections(result, names):
    """
    <summary>탐지 결과(xyxy, conf, cls)를 표 형태로 출력한다.</summary>
    <param name="result">results[0] (이미지 1장의 결과)</param>
    <param name="names">클래스 번호 → 이름 딕셔너리 (model.names)</param>
    <input>result.boxes의 xyxy, conf, cls 텐서</input>
    <output>없음 (콘솔 출력)</output>
    <reason>
    plot() 그림만 보면 내부 데이터 구조를 알 수 없다.
    박스 좌표, 신뢰도, 클래스 번호가 각각 어떤 배열에 들어 있는지 숫자로 확인해야
    직접 그리기와 STEP 7의 실시간 처리를 이해할 수 있다.
    </reason>
    """
    boxes = result.boxes

    # GPU 텐서일 수 있으므로 cpu()로 옮긴 뒤 numpy 배열로 바꾼다.
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    print()
    print(f"[탐지] 총 {len(boxes)}개 (conf ≥ {config.CONF_THRESHOLD})")

    if len(boxes) == 0:
        print("[안내] 탐지된 객체가 없습니다. 사람, 컵, 휴대폰 등이 보이게 다시 찍어 보세요.")
        return

    print(f"{'#':>2}  {'class':<14} {'id':>3}  {'conf':>5}  {'x1':>5} {'y1':>5} {'x2':>5} {'y2':>5}")
    print("-" * 58)

    for i, ((x1, y1, x2, y2), conf, cls) in enumerate(zip(xyxy, confs, classes)):
        print(f"{i:>2}  {names[cls]:<14} {cls:>3}  {conf:>5.2f}  {x1:>5.0f} {y1:>5.0f} {x2:>5.0f} {y2:>5.0f}")


def draw_detections(image, result, names):
    """
    <summary>탐지 박스와 "클래스명 신뢰도" 라벨을 cv2.rectangle, cv2.putText로 직접 그린다.</summary>
    <param name="image">그림을 그릴 BGR 이미지 (복사본을 만들어 그리므로 원본은 그대로)</param>
    <param name="result">results[0] (이미지 1장의 결과)</param>
    <param name="names">클래스 번호 → 이름 딕셔너리 (model.names)</param>
    <input>원본 이미지, 탐지 결과</input>
    <output>박스와 라벨이 그려진 새 이미지</output>
    <reason>
    plot()은 편리하지만 색, 글꼴, 표시 내용을 바꾸기 어렵다.
    STEP 5에서 익힌 그리기 함수로 직접 그리면 원하는 정보만 원하는 모양으로 표시할 수 있고,
    STEP 7의 실시간 화면도 같은 함수로 그린다.
    </reason>
    """
    canvas = image.copy()
    boxes = result.boxes

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
        color = BOX_COLORS[cls % len(BOX_COLORS)]
        label = f"{names[cls]} {conf:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=2)

        # 라벨 글자 크기를 재서 그 크기만큼 배경 상자를 채운다.
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 박스가 화면 맨 위에 붙어 있으면 라벨을 박스 안쪽에 그린다.
        label_top = y1 - text_h - baseline - 4

        if label_top < 0:
            label_top = y1

        label_bottom = label_top + text_h + baseline + 4

        cv2.rectangle(canvas, (x1, label_top), (x1 + text_w + 6, label_bottom), color, thickness=-1)
        cv2.putText(canvas, label, (x1 + 3, label_bottom - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_TEXT_COLOR, 1, cv2.LINE_AA)

    return canvas


def add_title(image, text):
    """
    <summary>이미지 위쪽에 제목 띠를 그린다.</summary>
    <param name="image">3채널 이미지 (직접 수정됨)</param>
    <param name="text">표시할 영문 제목</param>
    <input>비교용 패널 이미지</input>
    <output>제목이 그려진 같은 이미지</output>
    <reason>나란히 놓인 두 결과 중 어느 쪽이 직접 그린 것인지 화면만 보고 구분하게 한다.</reason>
    """
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (40, 40, 40), thickness=-1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)

    return image


def print_speed(result, elapsed_ms):
    """
    <summary>results[0].speed의 전처리·추론·후처리 시간과 비중을 출력한다.</summary>
    <param name="result">results[0] (이미지 1장의 결과)</param>
    <param name="elapsed_ms">predict 호출 전체를 perf_counter로 잰 시간 (ms)</param>
    <input>result.speed 딕셔너리 {"preprocess", "inference", "postprocess"} (단위 ms)</input>
    <output>없음 (콘솔 출력)</output>
    <reason>
    "YOLO가 느리다"고 할 때 실제로 어느 구간이 느린지 알아야 개선 방향을 정할 수 있다.
    CPU에서는 보통 추론(inference)이 대부분을 차지한다.
    세 구간 합과 전체 시간의 차이는 결과 객체 생성 등 ultralytics 내부 처리 시간이다.
    </reason>
    """
    speed = result.speed
    total = sum(speed.values())

    print()
    print("[속도] results[0].speed 구간별 시간")

    for key, label in (("preprocess", "전처리"), ("inference", "추론"), ("postprocess", "후처리")):
        share = speed[key] / total * 100 if total > 0 else 0
        print(f"  {label:<4} ({key:<11}) {speed[key]:7.1f} ms  {share:5.1f}%")

    print(f"  세 구간 합                {total:7.1f} ms")
    print(f"  predict 전체 (perf_counter) {elapsed_ms:7.1f} ms")


def compare_image_sizes(model, image, sizes, repeats):
    """
    <summary>입력 크기(imgsz)별로 추론 속도와 탐지 수를 비교해 표로 출력한다.</summary>
    <param name="model">YOLO 모델 객체</param>
    <param name="image">BGR 이미지 (numpy 배열)</param>
    <param name="sizes">비교할 입력 크기 목록 (예: (320, 416, 640))</param>
    <param name="repeats">크기마다 측정할 반복 횟수</param>
    <input>같은 이미지 1장</input>
    <output>크기별 결과 리스트 [(imgsz, 중앙값 ms, 탐지 수)]</output>
    <reason>
    imgsz가 작을수록 계산할 픽셀이 줄어 빨라지지만 작은 물체를 놓치기 쉽다.
    STEP 7 실시간 처리에서 속도와 정확도 사이의 적절한 크기를 고르는 근거로 쓴다.
    크기가 바뀌면 첫 추론에 준비 시간이 붙으므로 크기마다 워밍업 1회를 제외하고,
    측정값이 흔들리므로 평균 대신 중앙값을 쓴다.
    </reason>
    """
    rows = []

    print()
    print(f"[비교] imgsz별 속도·탐지 수 (각 {repeats}회 측정, 중앙값)")
    print(f"  {'imgsz':>5}  {'전체 ms':>8}  {'추론 ms':>8}  {'탐지 수':>6}")

    for size in sizes:
        # 워밍업: 결과는 버린다.
        run_predict(model, image, size)

        elapsed_list = []
        inference_list = []
        count = 0

        for _ in range(repeats):
            results, elapsed_ms = run_predict(model, image, size)
            elapsed_list.append(elapsed_ms)
            inference_list.append(results[0].speed["inference"])
            count = len(results[0].boxes)

        median_ms = statistics.median(elapsed_list)
        median_inference = statistics.median(inference_list)

        print(f"  {size:>5}  {median_ms:>8.1f}  {median_inference:>8.1f}  {count:>6}")

        rows.append((size, median_ms, count))

    return rows


def save_image(image, output_dir, name):
    """
    <summary>이미지를 outputs/ 폴더에 jpg로 저장한다.</summary>
    <param name="image">저장할 BGR 이미지</param>
    <param name="output_dir">저장 폴더 (Path)</param>
    <param name="name">파일 이름 앞부분</param>
    <input>비교 결과 이미지</input>
    <output>저장된 경로 (Path), 실패하면 None</output>
    <reason>Windows 한글 경로에서도 저장되도록 cv2.imwrite 대신 imencode + write_bytes를 쓴다.</reason>
    """
    output_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{name}_{stamp}.jpg"
    ok, jpeg = cv2.imencode(".jpg", image)

    if not ok:
        return None

    try:
        path.write_bytes(jpeg.tobytes())
    except OSError as error:
        print(f"[오류] 저장 실패: {error}")
        return None

    return path


def show_and_wait(title, image):
    """
    <summary>이미지를 창에 띄우고 키 입력 또는 창 닫기를 기다린다.</summary>
    <param name="title">창 제목 (영문 권장)</param>
    <param name="image">표시할 이미지 (numpy 배열)</param>
    <input>표시할 이미지, 키보드 입력</input>
    <output>없음</output>
    <reason>
    waitKey(0) 하나만 쓰면 창의 X 버튼으로 닫았을 때 멈추거나 오류가 날 수 있어
    짧게 기다리기를 반복하며 키 입력과 창 닫힘을 모두 확인한다.
    </reason>
    """
    print(f"[안내] '{title}' 창에서 아무 키나 누르면(또는 창을 닫으면) 종료합니다.")

    cv2.imshow(title, image)

    while True:
        if cv2.waitKey(100) != -1:
            break

        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break


def main():
    """
    <summary>캡처 이미지에 YOLO를 적용해 결과 표, 시각화 비교, 속도 분석을 실행한다.</summary>
    <input>captures/ 폴더의 최신 jpg 또는 웹캠 프레임, config.py 설정값</input>
    <output>콘솔 표 출력, 비교 창, outputs/ 폴더의 비교 이미지</output>
    <reason>
    실시간(STEP 7)으로 가기 전에 이미지 1장으로 "모델 불러오기 → 추론 → 결과 해석 → 그리기 → 속도 측정"을
    하나씩 확인해, 문제가 생겼을 때 어느 단계인지 구분할 수 있게 한다.
    </reason>
    """
    print(f"[정보] 추론 장치(DEVICE): {config.DEVICE}")

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

    print(f"[정보] 모델 불러오는 중: {config.MODEL_PATH}")
    model = load_model(config.MODEL_PATH, config.FALLBACK_MODEL_PATH)

    if model is None:
        print("[오류] 사용할 수 있는 YOLO 모델이 없어 종료합니다.")
        return

    try:
        # 워밍업: 첫 추론은 모델 준비(메모리 할당 등) 시간이 섞여 느리므로 측정에서 뺀다.
        _, warmup_ms = run_predict(model, image, config.IMG_SIZE)
        print(f"[정보] 워밍업 추론: {warmup_ms:.1f} ms (측정에서 제외)")

        results, elapsed_ms = run_predict(model, image, config.IMG_SIZE)
    except Exception as error:
        print(f"[오류] 추론 중 문제가 발생했습니다: {error}")
        print("[안내] DEVICE 설정이 이 컴퓨터에 맞는지 확인하세요. (GPU가 없으면 \"cpu\")")
        return

    result = results[0]

    print(f"[정보] 추론 시간 (imgsz={config.IMG_SIZE}): {elapsed_ms:.1f} ms")

    print_detections(result, model.names)
    print_speed(result, elapsed_ms)
    compare_image_sizes(model, image, COMPARE_SIZES, COMPARE_REPEATS)

    # plot()은 BGR numpy 배열을 돌려주므로 cv2 이미지와 바로 붙일 수 있다.
    manual = add_title(draw_detections(image, result, model.names), "Manual (cv2.rectangle / putText)")
    plotted = add_title(result.plot(), "results[0].plot()")
    comparison = cv2.hconcat([manual, plotted])

    saved = save_image(comparison, OUTPUT_DIR, "step6_compare")

    if saved is not None:
        print(f"\n[저장] {saved}")

    try:
        show_and_wait("Step6 - YOLO Image (Manual vs plot)", comparison)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# ============================================================
# 실행 방법
# ============================================================
# 1. 가상환경 활성화 (PowerShell)
#      .venv\Scripts\Activate.ps1
# 2. (권장) step3_capture.py로 사람, 컵, 휴대폰 등이 보이게 사진을 찍어 둔다.
# 3. 실행
#      python step6_yolo_image.py
# 4. 처음 실행하면 모델 파일(yolo26n.pt, 수 MB)을 자동으로 내려받는다. (인터넷 필요)
# 5. 비교 창에서 아무 키나 누르면 종료된다. 비교 이미지는 outputs/ 폴더에도 저장된다.
# 6. GPU가 있으면 config.py의 DEVICE를 "cuda:0"(NVIDIA) 또는 "mps"(Apple)로 바꿔 속도를 비교한다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. 전처리·추론·후처리 시간(results[0].speed) 구간별 비중
#    - preprocess : 이미지 크기 조정(letterbox), BGR→RGB, 0~1 정규화, 텐서 변환
#    - inference  : 신경망 계산 → CPU에서는 보통 전체의 대부분을 차지한다
#    - postprocess: 신뢰도 기준으로 거르고 원본 좌표로 되돌리기
#    - 추론이 대부분이면 "모델을 작게 하거나 imgsz를 줄이거나 GPU를 쓰는 것"이 가장 효과적이다.
# 2. 워밍업
#    - 첫 추론 시간과 두 번째 추론 시간을 비교한다. 첫 추론이 훨씬 느리다.
# 3. imgsz 320 / 416 / 640 비교
#    - 크기가 줄면 추론 시간이 대략 픽셀 수(크기의 제곱)에 비례해 줄어드는지 본다.
#    - 탐지 수가 줄어드는 크기가 있는지, 어떤 물체를 놓치는지 본다.
#    - STEP 5의 OpenCV 처리(수 ms)와 YOLO 추론 시간을 비교해 본다.
# 4. 결과 데이터 구조
#    - boxes.xyxy: [x1, y1, x2, y2] 왼쪽 위·오른쪽 아래 꼭짓점 (원본 이미지 픽셀 좌표)
#    - boxes.conf: 신뢰도 (0~1), boxes.cls: 클래스 번호 → model.names[번호]로 이름 변환
# 5. 직접 그리기 vs plot()
#    - 두 결과의 박스 위치가 같은지 확인한다 (같은 xyxy를 쓰므로 일치해야 한다).
#    - draw_detections의 색, 두께, 라벨 형식을 바꿔 본다.
