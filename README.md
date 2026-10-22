# 웹캠 실시간 YOLO 객체 인식 실습

웹캠 열기부터 브라우저 스트리밍, OpenCV 기초, YOLO 이미지 추론, 실시간 탐지, 파이프라인 대시보드까지
8단계로 따라 하는 파이썬 컴퓨터비전 실습 코드입니다.

- GPU 없는 노트북(CPU)에서 동작하도록 기본값을 맞췄습니다.
- 모든 함수와 클래스에 XML 형식 docstring(`<summary>`, `<param>`, `<input>`, `<output>`, `<reason>`)이 있습니다.
- 각 파일 맨 아래에 **실행 방법**과 **확인 포인트(학생이 관찰할 것)** 가 있습니다.

## 단계 구성

| 단계 | 파일 | 내용 |
|---|---|---|
| 1 | `step1_webcam.py` | 웹캠 열기, 실제 해상도·FPS 확인, 사용 가능한 카메라 번호 탐색 |
| 2 | `step2_server.py` | Flask MJPEG 스트리밍 (`/video_feed`), 카메라 스레드 + lock |
| 3 | `step3_capture.py` | 촬영 버튼 → `captures/`에 jpg 저장, 촬영 이력, 경로 조작 방지 |
| 4 | `step4_opencv_basics.py` | 이미지 정보, BGR/RGB/그레이 비교, `(x, y)`와 `[y, x]` 좌표 차이 |
| 5 | `step5_opencv_practice.py` | 키보드로 모드 전환(그레이·블러·Canny·이진화·HSV), FPS와 처리 시간 비교 |
| 6 | `step6_yolo_image.py` | 캡처 이미지에 YOLO 적용, 결과 표, 직접 그리기 vs `plot()`, imgsz별 속도 |
| 7 | `step7_realtime_yolo.py` | 실시간 YOLO (`--mode window` / `--mode web`), skip·imgsz·클래스 필터 |
| 8 | `step8_pipeline_dashboard.py` | 단계별 처리 시간을 플로우차트와 로그로 보여주는 웹 대시보드 |

앞 단계의 코드를 다음 단계가 가져다 쓰므로 모든 파일을 같은 폴더에 두고 실행합니다.

## 준비

Python 3.10 이상이 필요합니다.

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

- 6단계부터 쓰는 YOLO 모델(`yolo26n.pt`, 약 5MB)은 처음 실행할 때 자동으로 내려받습니다. 인터넷 연결이 필요합니다.
- PowerShell에서 "스크립트 실행이 차단됨" 오류가 나면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 한 번 실행합니다.

## 실행

```bash
python step1_webcam.py
python step2_server.py              # 브라우저에서 http://127.0.0.1:5000
python step3_capture.py             # 브라우저에서 http://127.0.0.1:5000
python step4_opencv_basics.py
python step5_opencv_practice.py
python step6_yolo_image.py
python step7_realtime_yolo.py               # OpenCV 창
python step7_realtime_yolo.py --mode web    # 브라우저
python step8_pipeline_dashboard.py          # 브라우저에서 http://127.0.0.1:5000
```

창 모드는 `q` 키 또는 창의 X 버튼으로, 서버는 터미널에서 `Ctrl+C`로 종료합니다.

## 설정

공통 설정은 `config.py`에 있습니다.

| 항목 | 기본값 | 설명 |
|---|---|---|
| `CAMERA_INDEX` | `0` | 웹캠 번호. 안 열리면 1단계가 사용 가능한 번호를 알려 줍니다. |
| `FRAME_WIDTH`, `FRAME_HEIGHT` | `640`, `480` | 요청 해상도 |
| `USE_DSHOW` | `False` | Windows에서 카메라가 늦게 켜지면 `True` |
| `MODEL_PATH` | `"yolo26n.pt"` | 불러오지 못하면 `FALLBACK_MODEL_PATH`(`yolo11n.pt`)로 다시 시도 |
| `IMG_SIZE` | `640` | 6단계 YOLO 입력 크기 |
| `CONF_THRESHOLD` | `0.4` | 이보다 낮은 신뢰도의 탐지는 버림 |
| `DEVICE` | `"cpu"` | `"cuda:0"`(NVIDIA GPU), `"mps"`(Apple Silicon) |

7·8단계 명령줄 옵션:

| 옵션 | 설명 |
|---|---|
| `--device cpu\|cuda:0\|mps` | 추론 장치. GPU가 없으면 경고 후 cpu로 실행 |
| `--imgsz N` | 입력 크기 (기본: CPU 320 / GPU 640) |
| `--skip N` | N프레임마다 1번 추론, 나머지는 직전 결과 재사용 (기본: CPU 2 / GPU 1) |
| `--conf 0.4` | 신뢰도 기준값 |
| `--classes person cup` | 지정한 클래스만 탐지 (공백이 있는 이름은 `"cell phone"`처럼 따옴표) |
| `--model 파일` | 모델 파일 |
| `--openvino` | OpenVINO 변환 모델로 실행 (Intel CPU 가속, 아래 참고) |
| `--threads N` | (8단계) CPU 추론 스레드 수, 기본 2, `0`이면 PyTorch 기본값 |

OpenVINO를 쓰려면 먼저 변환합니다.

```bash
pip install openvino
yolo export model=yolo26n.pt format=openvino imgsz=320
python step7_realtime_yolo.py --openvino
```

## 8단계 대시보드

화면 왼쪽은 탐지 영상과 촬영, 오른쪽 위는 플로우차트와 단계 상세, 오른쪽 아래는 로그 콘솔입니다.

- **플로우차트**: `capture → preprocess → inference → postprocess → visualize → output`.
  실행 중인 단계는 테두리로, 가장 느린 단계는 "병목"으로 표시하고, 각 노드 아래에 평균 ms와 비중(%)을 보여 줍니다.
  추론을 건너뛴 프레임에서는 inference 노드가 흐려집니다.
- **단계 상세**: 노드를 클릭하면 그 단계 함수의 docstring이 표시됩니다. 설명을 고치려면 코드의 docstring만 고치면 됩니다.
- **로그**: INFO(단계 통과), DEBUG(입출력 shape), WARN(프레임 드롭, 추론 생략). 일시정지와 레벨 필터가 있습니다.
- 데이터는 `GET /api/pipeline`(단계 정보)과 `GET /events`(Server-Sent Events, 0.2초마다 집계)로 받습니다.

## 알아 둘 점

- **카메라는 한 프로그램만 쓸 수 있습니다.** 2·3·7·8단계 서버, 다른 단계 창, Zoom·Teams 등을 동시에 켜지 마세요.
- **Windows에서 경로에 한글이 있으면** `cv2.imwrite`·`cv2.imread`가 오류 없이 실패합니다. 이 코드는 `cv2.imencode` + `write_bytes`, `np.fromfile` + `cv2.imdecode`로 우회합니다.
- **첫 YOLO 추론은 느립니다.** 모델 준비 때문에 첫 실행에서는 수 초 이상 걸릴 수 있으며, 속도 측정에서 뺍니다.
- **노트북은 전원을 연결하고 실행하세요.** 배터리로 동작하면 발열·전력 제한 때문에 추론 시간이 시간이 지날수록 커지고 들쭉날쭉해집니다.
- 촬영 사진은 `captures/`, 처리 결과는 `outputs/`에 저장되며 git에는 올라가지 않습니다(`.gitignore`).

## 라이선스 참고

YOLO 모델과 `ultralytics` 패키지는 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 라이선스입니다.
수업·학습 목적 사용은 문제없지만, 이 코드를 바탕으로 서비스를 배포한다면 해당 라이선스 조건을 확인하세요.
