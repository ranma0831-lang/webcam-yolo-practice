"""
8단계: 파이프라인 대시보드.

7단계 web 모드를 확장해, 영상과 함께 "지금 프레임이 어떤 단계를 지나고 있는지"를
플로우차트 강조와 로그로 실시간 확인한다.

단계: capture → preprocess → inference → postprocess → visualize → output
"""

import argparse
import atexit
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Response, jsonify, render_template

import config
import pipeline
from pipeline import WARN, recorder, stage
from step2_server import HOST, PORT, Camera, get_backend
from step3_capture import HISTORY_LIMIT, list_recent_captures
from step5_opencv_practice import FpsCounter
from step6_yolo_image import draw_detections
from step7_realtime_yolo import (
    apply_device_defaults,
    check_device,
    count_classes,
    create_web_app,
    draw_status,
    load_detector_model,
    resolve_class_ids,
)


# 촬영 이미지 저장 폴더 (3단계와 같은 폴더)
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"

# SSE로 집계 결과를 보내는 간격 (초)
SSE_INTERVAL = 0.2

# SSE 한 번에 보낼 최대 로그 줄 수
LOG_LINES_PER_TICK = 40

# FPS 이동평균에 사용할 최근 프레임 수
FPS_WINDOW = 30

# 추론하지 않는 프레임의 루프 간격 (카메라 약 30FPS에 맞춤)
IDLE_INTERVAL = 1 / 30

# 이 배수보다 프레임 간격이 길어지면 "프레임 드롭"으로 본다
DROP_FACTOR = 1.5


def limit_cpu_threads(device, threads):
    """
    <summary>CPU 추론에 쓰는 PyTorch 스레드 수를 제한한다.</summary>
    <param name="device">추론 장치 ("cpu"일 때만 적용)</param>
    <param name="threads">사용할 스레드 수 (0이면 PyTorch 기본값 유지)</param>
    <input>--threads 옵션 값</input>
    <output>실제로 설정된 스레드 수</output>
    <reason>
    PyTorch는 기본으로 거의 모든 코어에 스레드를 띄운다.
    노트북 CPU(고성능 코어 + 저전력 코어 혼합)에서는 느린 코어를 기다리느라 오히려 느려지고,
    대기하는 스레드가 CPU를 계속 써서 발열·전력 제한이 걸린다.
    그러면 같은 서버의 영상 전송과 브라우저까지 느려져 추론이 수백 ms로 들쭉날쭉해진다.
    이 실습 노트북 측정값: 기본 8스레드 39ms(CPU 7.2코어 사용) / 2스레드 32ms(1.9코어 사용).
    </reason>
    """
    if device == "cpu" and threads > 0:
        torch.set_num_threads(threads)

    return torch.get_num_threads()


def parse_args():
    """
    <summary>명령줄 옵션을 읽는다.</summary>
    <input>python step8_pipeline_dashboard.py --device cpu --imgsz 320 ... 형태의 인자</input>
    <output>argparse.Namespace (옵션 값 모음)</output>
    <reason>
    7단계와 같은 옵션을 유지해 같은 조건으로 비교할 수 있게 한다.
    8단계는 웹 대시보드 전용이므로 --mode 옵션은 없다.
    </reason>
    """
    parser = argparse.ArgumentParser(description="YOLO 파이프라인 대시보드")

    parser.add_argument("--device", choices=["cpu", "cuda:0", "mps"], default=config.DEVICE, help="추론 장치")
    parser.add_argument("--imgsz", type=int, default=None, help="입력 크기 (기본: CPU 320 / GPU 640)")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD, help="신뢰도 기준값 (0~1)")
    parser.add_argument("--skip", type=int, default=None, help="N프레임마다 1번 추론 (기본: CPU 2 / GPU 1)")
    parser.add_argument("--model", default=config.MODEL_PATH, help="모델 파일 (예: yolo26n.pt)")
    parser.add_argument("--openvino", action="store_true", help="OpenVINO 변환 모델로 실행 (Intel CPU 가속)")
    parser.add_argument("--classes", nargs="+", default=None, help="이 클래스만 탐지 (예: --classes person cup)")
    parser.add_argument("--threads", type=int, default=2, help="CPU 추론 스레드 수 (0: PyTorch 기본값)")

    return parser.parse_args()


# ============================================================
# 파이프라인 단계 함수 (docstring이 그대로 플로우차트 설명이 된다)
# ============================================================

@stage("capture", label="카메라 읽기")
def capture(camera):
    """
    <summary>카메라 스레드가 읽어 둔 최신 프레임 1장을 가져온다.</summary>
    <param name="camera">STEP 2의 Camera 객체</param>
    <input>웹캠 장치에서 계속 읽히고 있는 최신 프레임</input>
    <output>BGR 이미지 (480,640,3) uint8</output>
    <reason>
    카메라 읽기는 별도 스레드가 맡고 있어서 여기서는 복사만 하므로 매우 빠르다.
    파이프라인이 카메라보다 느리면 그사이 들어온 프레임은 덮어써져 사라지는데,
    이것이 로그의 WARN "프레임 드롭"이다.
    </reason>
    """
    return camera.get_frame()


@stage("preprocess", label="전처리", timing="speed")
def preprocess(result):
    """
    <summary>원본 이미지를 모델이 받는 텐서 모양으로 바꾼다 (ultralytics 내부).</summary>
    <param name="result">YOLO 결과 객체 (speed 값과 원본 이미지를 담고 있음)</param>
    <input>BGR 원본 이미지 (480,640,3) uint8</input>
    <output>모델 입력 텐서 (1,3,높이,너비) float, 값 범위 0~1</output>
    <reason>
    모델은 정해진 크기의 RGB 텐서만 받는다.
    그래서 비율을 유지한 채 imgsz에 맞게 줄이고 빈 곳을 채운 뒤(letterbox),
    BGR을 RGB로, 0~255를 0~1로, (높이,너비,채널)을 (채널,높이,너비)로 바꾼다.
    imgsz를 바꾸면 이 단계 출력 shape가 달라진다.
    </reason>
    """
    return result.orig_img, getattr(result, "tensor_shape", None)


@stage("inference", label="추론", timing="speed")
def inference(result):
    """
    <summary>신경망이 텐서를 계산해 후보 박스 숫자들을 만든다 (ultralytics 내부).</summary>
    <param name="result">YOLO 결과 객체</param>
    <input>모델 입력 텐서 (1,3,높이,너비)</input>
    <output>원시 예측 텐서 (후보 박스 좌표, 클래스별 점수)</output>
    <reason>
    YOLO 계산량의 대부분이 이 단계에 있어 CPU에서는 보통 가장 느린 단계(병목)가 된다.
    GPU는 이 계산을 병렬로 처리해 시간이 크게 줄어든다.
    입력 픽셀 수가 많을수록(imgsz가 클수록) 시간이 늘어난다.
    </reason>
    """
    return getattr(result, "tensor_shape", None), getattr(result, "raw_shape", None)


@stage("postprocess", label="후처리", timing="speed")
def postprocess(result):
    """
    <summary>후보 박스 중 신뢰도가 높은 것만 남기고 원본 이미지 좌표로 되돌린다 (ultralytics 내부).</summary>
    <param name="result">YOLO 결과 객체</param>
    <input>원시 예측 텐서</input>
    <output>최종 탐지 박스 목록 (xyxy, conf, cls)</output>
    <reason>
    모델은 수백 개의 후보를 내놓으므로 conf 기준값으로 거르고 겹치는 박스를 정리해야 한다.
    letterbox로 바뀐 좌표를 원본 크기로 되돌려야 원본 영상 위에 박스를 그릴 수 있다.
    </reason>
    """
    return getattr(result, "raw_shape", None), result


@stage("visualize", label="그리기")
def visualize(frame, result, names, fps, infer_ms, device_label):
    """
    <summary>탐지 박스, 라벨, FPS 등 상태 정보를 프레임 위에 그린다.</summary>
    <param name="frame">원본 BGR 프레임</param>
    <param name="result">가장 최근 YOLO 결과 (skip 프레임에서는 직전 결과)</param>
    <param name="names">클래스 번호와 이름 대응표 (model.names)</param>
    <param name="fps">이동평균 FPS</param>
    <param name="infer_ms">가장 최근 추론 시간</param>
    <param name="device_label">화면에 표시할 장치와 설정 요약</param>
    <input>BGR 원본 프레임과 탐지 결과</input>
    <output>박스와 정보가 그려진 BGR 이미지 (480,640,3) uint8</output>
    <reason>
    STEP 5에서 익힌 cv2.rectangle과 cv2.putText로 그린다.
    추론을 건너뛴 프레임에서도 직전 박스를 다시 그려 화면이 끊기지 않게 한다.
    </reason>
    """
    canvas = draw_detections(frame, result, names)
    counts = count_classes(result, names)
    draw_status(canvas, fps, infer_ms, counts, device_label)

    return canvas


@stage("output", label="출력")
def output(canvas, runner):
    """
    <summary>완성된 프레임을 웹 스트림(/detect_feed)이 가져갈 수 있게 공유한다.</summary>
    <param name="canvas">그리기가 끝난 BGR 이미지</param>
    <param name="runner">결과 프레임을 보관하는 PipelineRunner 객체</param>
    <input>박스가 그려진 BGR 이미지</input>
    <output>공유된 최신 프레임 (MJPEG 스트림과 촬영 버튼이 사용)</output>
    <reason>
    STEP 2와 같이 최신 1장만 lock으로 보호해 보관한다.
    JPEG 압축은 브라우저 연결마다 generate_frames 안에서 따로 일어난다.
    </reason>
    """
    runner.publish(canvas)

    return canvas


# ============================================================
# ultralytics 내부 계측
# ============================================================

def shape_of(value):
    """
    <summary>텐서 또는 텐서 리스트의 shape를 튜플로 돌려준다.</summary>
    <param name="value">torch 텐서, 또는 텐서가 담긴 리스트·튜플</param>
    <input>모델 입력 또는 원시 출력</input>
    <output>shape 튜플, 알 수 없으면 None</output>
    <reason>모델 종류에 따라 원시 출력이 텐서 하나일 수도, 여러 개의 묶음일 수도 있어 둘 다 처리한다.</reason>
    """
    if hasattr(value, "shape"):
        return tuple(value.shape)

    if isinstance(value, (list, tuple)) and value and hasattr(value[0], "shape"):
        return tuple(value[0].shape)

    return None


def instrument_predictor(model, probe):
    """
    <summary>ultralytics predictor의 전처리·추론·후처리 메서드를 감싸 활성 단계와 텐서 shape를 기록한다.</summary>
    <param name="model">워밍업 추론이 끝난 YOLO 모델 (predictor가 만들어져 있어야 함)</param>
    <param name="probe">텐서 shape를 담아 둘 딕셔너리 (직접 수정됨)</param>
    <input>model.predictor</input>
    <output>계측에 성공하면 True</output>
    <reason>
    시간은 results[0].speed로 알 수 있지만, "지금 어느 단계가 실행 중인지"와
    단계 사이를 오가는 텐서 shape는 predict 바깥에서 볼 수 없다.
    predictor의 메서드를 감싸면 모델 코드를 고치지 않고도 이 정보를 얻을 수 있다.
    ultralytics 버전이 바뀌어 메서드가 없으면 계측만 생략하고 나머지는 정상 동작한다.
    </reason>
    """
    predictor = getattr(model, "predictor", None)

    if predictor is None or not all(hasattr(predictor, name) for name in ("preprocess", "inference", "postprocess")):
        print("[경고] ultralytics predictor를 찾지 못해 내부 단계 shape 계측을 생략합니다.")
        return False

    original_preprocess = predictor.preprocess
    original_inference = predictor.inference
    original_postprocess = predictor.postprocess

    def preprocess_hook(*args, **kwargs):
        recorder.set_active("preprocess")
        tensor = original_preprocess(*args, **kwargs)
        probe["tensor_shape"] = shape_of(tensor)
        return tensor

    def inference_hook(*args, **kwargs):
        recorder.set_active("inference")
        preds = original_inference(*args, **kwargs)
        probe["raw_shape"] = shape_of(preds)
        return preds

    def postprocess_hook(*args, **kwargs):
        recorder.set_active("postprocess")
        return original_postprocess(*args, **kwargs)

    predictor.preprocess = preprocess_hook
    predictor.inference = inference_hook
    predictor.postprocess = postprocess_hook

    return True


def run_model(model, frame, options, probe):
    """
    <summary>YOLO 추론을 실행하고, 단계 기록에 필요한 시각과 shape를 결과 객체에 붙인다.</summary>
    <param name="model">YOLO 모델</param>
    <param name="frame">BGR 프레임</param>
    <param name="options">predict 옵션 딕셔너리</param>
    <param name="probe">instrument_predictor가 채우는 shape 딕셔너리</param>
    <input>원본 프레임</input>
    <output>results[0] 객체 (started_at, tensor_shape, raw_shape 속성 추가)</output>
    <reason>
    model.predict() 한 번 안에 전처리·추론·후처리 세 단계가 숨어 있다.
    시작 시각을 기록해 두면 speed 값으로 각 단계의 시작과 끝을 다시 계산할 수 있다.
    </reason>
    """
    started_at = time.perf_counter()
    result = model.predict(source=frame, **options)[0]

    result.started_at = started_at
    result.tensor_shape = probe.get("tensor_shape")
    result.raw_shape = probe.get("raw_shape")

    return result


class PipelineRunner:
    """
    <summary>캡처부터 출력까지 파이프라인을 별도 스레드에서 반복 실행하는 클래스.</summary>
    <param name="camera">STEP 2의 Camera 객체</param>
    <param name="model">YOLO 모델</param>
    <param name="options">predict 옵션 딕셔너리</param>
    <param name="skip">N프레임마다 1번 추론</param>
    <param name="device_label">화면에 표시할 장치와 설정 요약</param>
    <param name="probe">텐서 shape 딕셔너리</param>
    <input>카메라 프레임</input>
    <output>get_frame()으로 결과 프레임을, recorder로 단계 이벤트를 제공한다.</output>
    <reason>
    STEP 7의 DetectionStream과 같은 구조(별도 스레드 + 최신 1장 + lock)에
    단계 함수 호출과 이벤트 기록을 더했다.
    get_frame()과 running이 Camera와 같아 STEP 2의 generate_frames를 그대로 쓸 수 있다.
    </reason>
    """

    def __init__(self, camera, model, options, skip, device_label, probe):
        """
        <summary>설정을 저장하고 파이프라인 스레드를 시작한다.</summary>
        <param name="camera">Camera 객체</param>
        <param name="model">YOLO 모델</param>
        <param name="options">predict 옵션</param>
        <param name="skip">추론 간격</param>
        <param name="device_label">장치와 설정 요약</param>
        <param name="probe">텐서 shape 딕셔너리</param>
        <input>없음</input>
        <output>없음</output>
        <reason>카메라가 알려주는 FPS로 "정상 프레임 간격"을 정해 프레임 드롭 판단 기준으로 쓴다.</reason>
        """
        self.camera = camera
        self.model = model
        self.options = options
        self.skip = skip
        self.device_label = device_label
        self.probe = probe

        camera_fps = camera.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1 / camera_fps if camera_fps > 0 else 1 / 30

        self.fps_counter = FpsCounter(FPS_WINDOW)
        self.fps = 0.0
        self.last_result = None
        self.last_infer_ms = 0.0
        self.last_capture_time = None

        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    @property
    def running(self):
        """
        <summary>카메라와 파이프라인 스레드가 모두 동작 중인지 알려준다.</summary>
        <input>없음</input>
        <output>bool</output>
        <reason>generate_frames와 SSE가 스트림 종료를 판단하는 기준이다.</reason>
        """
        return self.camera.running and not self.stopped

    def _check_frame_drop(self, now):
        """
        <summary>직전 캡처와의 간격으로 놓친 카메라 프레임 수를 추정해 WARN을 남긴다.</summary>
        <param name="now">이번 캡처 시각 (perf_counter 초)</param>
        <input>캡처 간격</input>
        <output>없음</output>
        <reason>
        카메라는 약 33ms마다 새 프레임을 만들지만 파이프라인이 그보다 오래 걸리면
        중간 프레임은 처리되지 못하고 사라진다.
        간격을 정상 간격으로 나누면 대략 몇 장을 놓쳤는지 알 수 있다.
        </reason>
        """
        if self.last_capture_time is not None:
            gap = now - self.last_capture_time

            if gap > self.frame_interval * DROP_FACTOR:
                missed = round(gap / self.frame_interval) - 1
                recorder.record("capture", WARN, f"프레임 드롭: 약 {missed}장 놓침 (간격 {gap * 1000:.0f}ms)")

        self.last_capture_time = now

    def _update(self):
        """
        <summary>파이프라인 스레드 본체. 프레임마다 6단계를 차례로 실행한다.</summary>
        <input>카메라 프레임</input>
        <output>없음 (결과 프레임 공유, 이벤트 기록)</output>
        <reason>
        skip 간격이 아닌 프레임은 전처리·추론·후처리를 건너뛰고 직전 결과를 재사용하며,
        그 사실을 WARN으로 남겨 skip 값과 로그의 관계를 볼 수 있게 한다.
        오류가 나면 원인을 출력하고 멈춰 브라우저가 멈춘 화면을 계속 받지 않게 한다.
        </reason>
        """
        while self.running:
            frame_index = recorder.begin_frame()
            captured_at = time.perf_counter()
            frame = capture(self.camera)

            if frame is None:
                recorder.set_active(None)
                time.sleep(0.01)
                continue

            self._check_frame_drop(captured_at)

            should_infer = self.last_result is None or (frame_index - 1) % self.skip == 0

            try:
                if should_infer:
                    result = run_model(self.model, frame, self.options, self.probe)
                    preprocess(result)
                    inference(result)
                    postprocess(result)

                    self.last_result = result
                    self.last_infer_ms = sum(result.speed.values())
                    recorder.set_skipped(False)
                else:
                    reused = len(self.last_result.boxes)
                    recorder.record("inference", WARN, f"skip={self.skip}: 추론 생략, 직전 결과({reused} boxes) 재사용")
                    recorder.set_skipped(True)

                self.fps = self.fps_counter.tick()
                canvas = visualize(frame, self.last_result, self.model.names, self.fps, self.last_infer_ms, self.device_label)
                output(canvas, self)

            except Exception as error:
                print(f"[오류] 파이프라인 처리 중 문제가 발생했습니다: {error}")
                self.stopped = True
                break

            recorder.set_active(None)

            if not should_infer:
                time.sleep(IDLE_INTERVAL)

    def publish(self, canvas):
        """
        <summary>결과 프레임을 공유 변수에 저장한다.</summary>
        <param name="canvas">완성된 BGR 이미지</param>
        <input>output 단계의 이미지</input>
        <output>없음</output>
        <reason>파이프라인 스레드(쓰기)와 웹 요청 스레드(읽기)가 겹치지 않게 lock을 쓴다.</reason>
        """
        with self.lock:
            self.frame = canvas

    def get_frame(self):
        """
        <summary>최신 결과 프레임의 복사본을 돌려준다.</summary>
        <input>공유 프레임</input>
        <output>numpy 배열 (아직 없으면 None)</output>
        <reason>Camera.get_frame과 같은 방식이라 스트림과 촬영 코드를 그대로 재사용한다.</reason>
        """
        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):
        """
        <summary>파이프라인 스레드를 멈춘다.</summary>
        <input>없음</input>
        <output>없음</output>
        <reason>카메라를 해제하기 전에 먼저 멈춰 해제된 카메라에 접근하지 않게 한다.</reason>
        """
        self.stopped = True

        if self.thread.is_alive():
            self.thread.join(timeout=2)


# ============================================================
# 웹 서버
# ============================================================

def build_snapshot(runner, settings, last_seq):
    """
    <summary>최근 이벤트를 집계해 대시보드에 보낼 데이터 1건을 만든다.</summary>
    <param name="runner">PipelineRunner 객체</param>
    <param name="settings">상단 바에 표시할 설정 딕셔너리</param>
    <param name="last_seq">이 연결이 마지막으로 받은 이벤트 일련번호</param>
    <input>recorder의 링버퍼와 상태</input>
    <output>(보낼 데이터 딕셔너리, 새 last_seq)</output>
    <reason>
    매 프레임 이벤트를 그대로 보내면 1초에 수백 건이 되어 CPU 노트북에 부담이 크다.
    0.2초마다 단계별 평균 ms, 활성 단계, 병목, 최근 로그 몇 줄만 묶어 보낸다.
    비중(%)은 각 단계를 "한 번 지날 때" 걸리는 평균 시간끼리 비교한 값이다.
    </reason>
    """
    events = recorder.events_since(last_seq)
    new_seq = events[-1]["seq"] if events else last_seq

    averages = recorder.stage_averages()
    active, skipped, frame_index = recorder.state()
    total = sum(averages.values())
    bottleneck = max(averages, key=averages.get) if averages else None

    logs = [
        {"level": event["level"], "text": pipeline.format_event(event)}
        for event in events[-LOG_LINES_PER_TICK:]
    ]

    payload = {
        "running": runner.running,
        "frame": frame_index,
        "fps": round(runner.fps, 1),
        "active": active,
        "skipped": skipped,
        "stages": {stage_id: round(ms, 2) for stage_id, ms in averages.items()},
        "total_ms": round(total, 2),
        "bottleneck": bottleneck,
        "logs": logs,
        "dropped_logs": max(0, len(events) - LOG_LINES_PER_TICK),
        **settings,
    }

    return payload, new_seq


def create_dashboard_app(camera, runner, settings):
    """
    <summary>STEP 7 웹 앱에 대시보드 페이지, 단계 정보 API, SSE 이벤트 경로를 더한다.</summary>
    <param name="camera">원본 프레임 Camera 객체</param>
    <param name="runner">결과 프레임 PipelineRunner 객체</param>
    <param name="settings">상단 바에 표시할 설정 딕셔너리</param>
    <input>Camera, PipelineRunner, 설정</input>
    <output>설정이 끝난 Flask 앱</output>
    <reason>
    STEP 7의 /video_feed, /detect_feed, /capture, /captures 경로는 그대로 재사용하고,
    첫 페이지(/)만 대시보드로 바꾼다.
    Flask는 경로 이름(endpoint)마다 처리 함수를 view_functions에 보관하므로 그 항목만 교체한다.
    </reason>
    """
    app = create_web_app(camera, runner, CAPTURE_DIR)

    def dashboard():
        """
        <summary>대시보드 페이지를 돌려준다.</summary>
        <input>GET /</input>
        <output>templates/dashboard.html을 렌더링한 HTML</output>
        <reason>첫 화면에 촬영 이력과 설정 값을 서버에서 바로 채워 넣는다.</reason>
        """
        return render_template(
            "dashboard.html",
            settings=settings,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            captures=list_recent_captures(CAPTURE_DIR, HISTORY_LIMIT),
            history_limit=HISTORY_LIMIT,
        )

    app.view_functions["index"] = dashboard

    @app.route("/api/pipeline")
    def api_pipeline():
        """
        <summary>파이프라인 단계 목록과 docstring 설명을 JSON으로 돌려준다.</summary>
        <input>GET /api/pipeline</input>
        <output>[{id, label, summary, input, output, reason, params}]</output>
        <reason>브라우저가 이 데이터로 플로우차트를 만들기 때문에 단계가 바뀌어도 HTML을 고칠 필요가 없다.</reason>
        """
        return jsonify(pipeline.pipeline_metadata())

    @app.route("/events")
    def events():
        """
        <summary>Server-Sent Events로 0.2초마다 집계 데이터를 브라우저에 보낸다.</summary>
        <input>GET /events (브라우저의 EventSource가 연결을 유지)</input>
        <output>text/event-stream 응답 ("data: JSON" 줄이 계속 이어짐)</output>
        <reason>
        WebSocket 대신 SSE를 쓰는 이유:
        서버에서 브라우저로 보내기만 하면 되는 단방향 통신이고,
        Flask의 스트리밍 응답(제너레이터)만으로 구현되어 추가 라이브러리가 필요 없으며,
        브라우저 EventSource가 연결이 끊기면 자동으로 다시 연결해 준다.
        </reason>
        """
        def stream():
            last_seq = 0

            while True:
                payload, last_seq = build_snapshot(runner, settings, last_seq)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                time.sleep(SSE_INTERVAL)

        return Response(stream(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app


def main():
    """
    <summary>옵션 정리, 모델 준비, 카메라와 파이프라인 시작, 대시보드 서버 실행을 차례로 한다.</summary>
    <input>명령줄 옵션, config.py 설정값</input>
    <output>http://127.0.0.1:5000 대시보드</output>
    <reason>
    STEP 7과 같은 준비 과정을 쓰되, 워밍업 뒤에 ultralytics 내부 계측을 붙인다.
    predictor는 첫 추론 때 만들어지므로 워밍업이 끝난 뒤에야 감쌀 수 있다.
    </reason>
    """
    args = parse_args()
    args.device = check_device(args.device)
    apply_device_defaults(args)

    threads = limit_cpu_threads(args.device, args.threads)

    model, backend_name = load_detector_model(args)

    if model is None:
        print("[오류] 사용할 수 있는 YOLO 모델이 없어 종료합니다.")
        return

    try:
        class_ids = resolve_class_ids(args.classes, model.names)
    except ValueError as error:
        print(f"[오류] {error}")
        return

    options = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device,
        "classes": class_ids,
        "verbose": False,
    }

    # half는 CUDA에서만 넘긴다. (ultralytics 8.4부터 폐지 예정 경고가 있어 CPU에서는 생략)
    if args.half:
        options["half"] = True

    print("[정보] 모델 워밍업 중... (처음 한 번은 몇 초 걸릴 수 있습니다)")

    try:
        warmup_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        model.predict(source=warmup_frame, **options)
    except Exception as error:
        print(f"[오류] 워밍업 추론에 실패했습니다: {error}")
        print("[안내] --device 설정이 이 컴퓨터에 맞는지 확인하세요.")
        return

    probe = {}
    instrument_predictor(model, probe)

    device_label = f"{args.device} ({backend_name}) | imgsz {args.imgsz} | skip {args.skip} | conf {args.conf}"

    if args.device == "cpu":
        device_label += f" | threads {threads}"

    settings = {
        "device": args.device,
        "backend": backend_name,
        "threads": threads if args.device == "cpu" else None,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "skip": args.skip,
        "half": args.half,
        "classes": args.classes or [],
    }

    print(f"[설정] {device_label}")

    CAPTURE_DIR.mkdir(exist_ok=True)

    try:
        camera = Camera(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT, get_backend(config.USE_DSHOW))
    except RuntimeError as error:
        print(f"[오류] {error}")
        print("[안내] step1_webcam.py로 사용 가능한 카메라 번호를 확인하고 config.py의 CAMERA_INDEX를 바꿔 보세요.")
        print("[안내] 다른 프로그램(step2/3/7 서버, Zoom 등)이 카메라를 쓰고 있지 않은지도 확인하세요.")
        return

    runner = PipelineRunner(camera, model, options, args.skip, device_label, probe)

    # atexit은 나중에 등록한 것부터 실행한다 → runner.stop() 후 camera.release()
    atexit.register(camera.release)
    atexit.register(runner.stop)

    app = create_dashboard_app(camera, runner, settings)

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
# 2. 실행 (CPU 권장값 imgsz=320, skip=2 자동 적용)
#      python step8_pipeline_dashboard.py
# 3. 브라우저에서 http://127.0.0.1:5000 접속
# 4. 옵션 예시
#      python step8_pipeline_dashboard.py --imgsz 640 --skip 1
#      python step8_pipeline_dashboard.py --skip 4
#      python step8_pipeline_dashboard.py --device cuda:0      (NVIDIA GPU)
#      python step8_pipeline_dashboard.py --threads 0          (PyTorch 기본 스레드 수로 비교)
# 5. 플로우차트 노드를 클릭하면 그 단계 함수의 docstring 설명이 오른쪽에 나온다.
# 6. step2/3/7 서버나 다른 카메라 프로그램이 켜져 있으면 먼저 끈다.
# 7. 노트북은 전원을 연결하고 실행한다.
#    배터리로 동작하면 발열·전력 제한 때문에 inference ms가 시간이 지날수록 커지고
#    수십~수백 ms로 들쭉날쭉해져 단계별 비교가 어렵다.
#
# ============================================================
# 확인 포인트 (학생이 관찰할 것)
# ============================================================
# 1. CPU에서 inference가 전체 시간의 몇 %인가?
#    - 플로우차트 노드 아래의 "ms · %"를 읽는다. 병목 표시가 어느 노드에 붙는지 본다.
#    - preprocess와 postprocess도 0이 아니다. "모델 밖" 처리도 시간이 든다는 점을 확인한다.
# 2. imgsz를 바꾸면 어느 노드의 숫자가 변하는가?
#    - --imgsz 320과 640으로 각각 실행해 노드별 ms를 비교한다.
#    - DEBUG 로그에서 preprocess의 out=(1,3,높이,너비) shape가 바뀌는 것을 확인한다.
#    - capture, visualize, output은 원본 해상도(640x480)를 다루므로 거의 변하지 않는다.
# 3. skip 값에 따라 로그의 WARN 빈도가 어떻게 바뀌는가?
#    - --skip 1, 2, 4로 실행해 "추론 생략" WARN이 몇 줄마다 나오는지 센다.
#    - skip=1이면 추론 생략 WARN은 없지만, 프레임 드롭 WARN이 늘어날 수 있다.
#      (매 프레임 추론하느라 카메라 속도를 따라가지 못하기 때문)
#    - 추론을 건너뛴 프레임에서는 inference 노드가 흐려지는 것을 확인한다.
# 4. CPU vs GPU
#    - GPU에서 실행하면 inference 노드의 ms가 크게 줄고, 병목이 다른 노드로 옮겨갈 수 있다.
#    - 이때 상단 바의 장치 표시와 노드별 ms를 CPU 결과와 나란히 기록한다.
# 5. CPU 스레드 수 (--threads)
#    - --threads 0(기본값, 거의 모든 코어)과 --threads 2로 각각 1분 이상 실행해 inference ms를 비교한다.
#    - 작업 관리자에서 python의 CPU 사용률도 함께 본다.
#    - "스레드를 많이 쓸수록 빠르다"가 항상 맞지 않는다는 것, 특히 배터리로 도는 노트북에서는
#      발열·전력 제한 때문에 시간이 지날수록 느려질 수 있다는 것을 확인한다.
# 6. 코드와 화면의 연결
#    - pipeline.py의 @stage 데코레이터와 이 파일의 단계 함수 docstring을 고치고 재실행하면
#      플로우차트 설명이 바뀌는 것을 확인한다.
