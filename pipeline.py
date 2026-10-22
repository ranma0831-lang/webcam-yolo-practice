"""
STEP 8 파이프라인 계측 모듈.

각 단계 함수에 @stage("단계이름") 데코레이터를 붙이면
실행 시간과 입력·출력 요약이 이벤트로 기록되고,
함수의 XML docstring은 웹 대시보드의 플로우차트 설명으로 쓰인다.
"""

import functools
import inspect
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime


# 파이프라인 단계 순서 (플로우차트도 이 순서로 그린다)
STAGE_ORDER = ("capture", "preprocess", "inference", "postprocess", "visualize", "output")

# ultralytics model.predict() 한 번 안에서 일어나는 단계
SPEED_STAGES = ("preprocess", "inference", "postprocess")

# 이벤트 링버퍼 크기
EVENT_BUFFER_SIZE = 500

# 단계별 평균을 낼 때 사용할 최근 기록 수
AVERAGE_WINDOW = 30

# 로그 레벨
INFO = "INFO"
DEBUG = "DEBUG"
WARN = "WARN"

# 등록된 단계: 단계 id → {"id", "label", "func"}
_registry = {}


def describe(value):
    """
    <summary>로그에 적을 수 있도록 값을 짧은 문자열로 요약한다.</summary>
    <param name="value">요약할 값 (numpy 배열, 텐서, shape 튜플, YOLO 결과, 문자열 등)</param>
    <input>단계 함수의 입력 또는 출력</input>
    <output>"(480,640,3) uint8", "5 boxes" 같은 요약 문자열</output>
    <reason>
    이미지 배열을 그대로 로그에 찍으면 숫자가 수십만 개라 읽을 수 없다.
    학생이 알아야 하는 것은 "모양(shape)과 자료형(dtype)", "탐지 개수"이므로 그것만 남긴다.
    </reason>
    """
    if value is None:
        return "-"

    if isinstance(value, str):
        return value

    # YOLO 결과 객체는 boxes를 가지고 있다.
    boxes = getattr(value, "boxes", None)

    if boxes is not None:
        return f"{len(boxes)} boxes"

    # numpy 배열, torch 텐서
    shape = getattr(value, "shape", None)

    if shape is not None:
        dtype = str(getattr(value, "dtype", "")).replace("torch.", "")
        return f"({','.join(str(n) for n in shape)}) {dtype}".strip()

    # shape 튜플 (예: (1, 3, 320, 320))
    if isinstance(value, tuple) and all(isinstance(n, int) for n in value):
        return f"({','.join(str(n) for n in value)})"

    return type(value).__name__


class PipelineRecorder:
    """
    <summary>파이프라인 이벤트를 링버퍼에 모으고, 집계 값을 계산해 주는 기록기.</summary>
    <param name="maxlen">링버퍼에 보관할 최대 이벤트 수</param>
    <input>단계 데코레이터와 파이프라인 스레드가 보내는 이벤트</input>
    <output>events_since(), stage_averages(), state()로 기록을 읽어 갈 수 있다.</output>
    <reason>
    파이프라인 스레드는 이벤트를 쓰고, 웹 요청 스레드(SSE)는 동시에 읽는다.
    deque(maxlen)은 오래된 이벤트를 자동으로 버려 메모리가 계속 늘지 않고,
    lock으로 읽기와 쓰기가 겹치지 않게 보호한다.
    </reason>
    """

    def __init__(self, maxlen):
        """
        <summary>빈 링버퍼와 상태 값을 준비한다.</summary>
        <param name="maxlen">링버퍼 크기</param>
        <input>없음</input>
        <output>없음</output>
        <reason>seq(일련번호)를 두어 SSE가 "이미 보낸 이벤트 이후"만 골라 보낼 수 있게 한다.</reason>
        """
        self.events = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.seq = 0
        self.frame_index = 0
        self.active_stage = None
        self.skipped = False

    def begin_frame(self):
        """
        <summary>새 프레임 처리를 시작하고 프레임 번호를 1 늘린다.</summary>
        <input>없음</input>
        <output>새 프레임 번호 (int)</output>
        <reason>모든 이벤트에 프레임 번호를 붙여 "몇 번째 프레임이 어느 단계를 지났는지" 추적한다.</reason>
        """
        with self.lock:
            self.frame_index += 1
            return self.frame_index

    def set_active(self, stage_id):
        """
        <summary>지금 실행 중인 단계를 기록한다.</summary>
        <param name="stage_id">실행 중인 단계 id (없으면 None)</param>
        <input>단계 시작 시점의 호출</input>
        <output>없음</output>
        <reason>대시보드가 0.2초마다 이 값을 읽어 플로우차트에서 현재 단계를 강조한다.</reason>
        """
        with self.lock:
            self.active_stage = stage_id

    def set_skipped(self, skipped):
        """
        <summary>가장 최근 프레임이 추론을 건너뛰었는지 기록한다.</summary>
        <param name="skipped">추론을 건너뛰었으면 True</param>
        <input>파이프라인 스레드의 skip 판단</input>
        <output>없음</output>
        <reason>대시보드가 이 값으로 inference 노드를 흐리게 표시한다.</reason>
        """
        with self.lock:
            self.skipped = skipped

    def record(self, stage_id, level, message, ms=None, start=None, end=None):
        """
        <summary>이벤트 1개를 링버퍼에 추가한다.</summary>
        <param name="stage_id">단계 id</param>
        <param name="level">로그 레벨 (INFO, DEBUG, WARN)</param>
        <param name="message">로그에 보일 내용</param>
        <param name="ms">소요 시간 (밀리초, 없으면 None)</param>
        <param name="start">시작 시각 (perf_counter 초)</param>
        <param name="end">종료 시각 (perf_counter 초)</param>
        <input>단계 실행 결과</input>
        <output>없음</output>
        <reason>
        시각은 두 가지로 남긴다.
        perf_counter는 구간 길이를 정확히 재는 데, 벽시계 시각(time)은 사람이 로그를 읽는 데 쓴다.
        </reason>
        """
        with self.lock:
            self.seq += 1
            self.events.append({
                "seq": self.seq,
                "frame": self.frame_index,
                "stage": stage_id,
                "level": level,
                "message": message,
                "ms": ms,
                "start": start,
                "end": end,
                "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            })

    def log_pass(self, stage_id, ms, start, end, input_desc, output_desc):
        """
        <summary>단계 통과를 INFO(소요 시간)와 DEBUG(입출력 요약) 이벤트 2개로 기록한다.</summary>
        <param name="stage_id">단계 id</param>
        <param name="ms">소요 시간 (밀리초)</param>
        <param name="start">시작 시각 (perf_counter 초)</param>
        <param name="end">종료 시각 (perf_counter 초)</param>
        <param name="input_desc">입력 요약 문자열</param>
        <param name="output_desc">출력 요약 문자열</param>
        <input>데코레이터가 측정한 값</input>
        <output>없음</output>
        <reason>
        시간과 shape를 레벨로 나눠 두면 대시보드에서 DEBUG를 끄고 시간 흐름만 볼 수도 있다.
        </reason>
        """
        self.record(stage_id, INFO, f"{ms:.1f}ms", ms=ms, start=start, end=end)
        self.record(stage_id, DEBUG, f"in={input_desc} out={output_desc}")

    def events_since(self, seq):
        """
        <summary>일련번호 seq 이후에 기록된 이벤트만 돌려준다.</summary>
        <param name="seq">마지막으로 받은 이벤트 일련번호</param>
        <input>링버퍼</input>
        <output>이벤트 딕셔너리 리스트 (오래된 것부터)</output>
        <reason>SSE 연결마다 이미 보낸 로그를 다시 보내지 않기 위해서다.</reason>
        """
        with self.lock:
            return [event for event in self.events if event["seq"] > seq]

    def stage_averages(self):
        """
        <summary>단계별 최근 소요 시간의 평균을 계산한다.</summary>
        <input>링버퍼의 INFO 이벤트 중 ms 값이 있는 것</input>
        <output>{단계 id: 평균 ms} 딕셔너리 (기록이 없는 단계는 빠짐)</output>
        <reason>
        한 프레임 값은 흔들림이 커서 병목을 판단하기 어렵다.
        단계마다 최근 AVERAGE_WINDOW개를 평균 내 안정된 숫자를 보여준다.
        </reason>
        """
        with self.lock:
            events = list(self.events)

        samples = {}

        for event in events:
            if event["level"] == INFO and event["ms"] is not None:
                samples.setdefault(event["stage"], []).append(event["ms"])

        return {
            stage_id: sum(values[-AVERAGE_WINDOW:]) / len(values[-AVERAGE_WINDOW:])
            for stage_id, values in samples.items()
        }

    def state(self):
        """
        <summary>현재 활성 단계, skip 여부, 프레임 번호를 한 번에 읽는다.</summary>
        <input>기록기 상태 값</input>
        <output>(active_stage, skipped, frame_index) 튜플</output>
        <reason>세 값을 lock 한 번으로 읽어 서로 어긋나지 않게 한다.</reason>
        """
        with self.lock:
            return self.active_stage, self.skipped, self.frame_index


# 모든 단계가 함께 쓰는 기록기 (파이프라인 스레드 1개가 쓰고, SSE가 읽는다)
recorder = PipelineRecorder(EVENT_BUFFER_SIZE)


def format_event(event):
    """
    <summary>이벤트를 로그 콘솔에 표시할 한 줄 문자열로 만든다.</summary>
    <param name="event">record()가 만든 이벤트 딕셔너리</param>
    <input>이벤트 1개</input>
    <output>"[14:40:02.135] #frame 0421 | inference | 38.2ms" 형식의 문자열</output>
    <reason>서버에서 형식을 정해 보내면 브라우저는 그대로 표시만 하면 된다.</reason>
    """
    return f"[{event['time']}] #frame {event['frame']:04d} | {event['stage']} | {event['message']}"


def stage(stage_id, label=None, timing="wall"):
    """
    <summary>함수를 파이프라인 단계로 등록하고, 실행할 때마다 이벤트를 기록하는 데코레이터.</summary>
    <param name="stage_id">단계 id (STAGE_ORDER 중 하나)</param>
    <param name="label">플로우차트에 보일 이름 (없으면 stage_id)</param>
    <param name="timing">"wall"이면 함수 실행 시간을 직접 재고, "speed"이면 YOLO 결과의 speed 값을 쓴다.</param>
    <input>꾸밀 단계 함수</input>
    <output>기록 기능이 더해진 같은 이름의 함수</output>
    <reason>
    단계 함수마다 시간 측정 코드를 반복해 쓰면 본래 하는 일이 잘 안 보인다.
    데코레이터로 측정과 기록을 한 곳에 모으면 단계 함수는 자기 일만 하면 된다.

    전처리·추론·후처리는 model.predict() 한 번 호출 안에 숨어 있어 바깥에서 따로 잴 수 없다.
    그래서 timing="speed" 단계는 함수 실행 시간 대신 results[0].speed의 해당 값을 쓰고,
    시작 시각은 predict 시작 시각에 앞 단계 시간을 더해 계산한다.
    이 경우 단계 함수는 (입력 값, 출력 값) 튜플을 돌려준다.
    </reason>
    """
    if stage_id not in STAGE_ORDER:
        raise ValueError(f"알 수 없는 단계 id: {stage_id} (사용 가능: {STAGE_ORDER})")

    def decorator(func):
        """
        <summary>단계 함수를 레지스트리에 등록하고 기록용 래퍼로 감싼다.</summary>
        <param name="func">단계 함수</param>
        <input>단계 함수</input>
        <output>래퍼 함수</output>
        <reason>functools.wraps로 원래 이름과 docstring을 유지해 플로우차트 설명에 쓸 수 있게 한다.</reason>
        """
        _registry[stage_id] = {"id": stage_id, "label": label or stage_id, "func": func}

        if timing == "speed":
            index = SPEED_STAGES.index(stage_id)

            @functools.wraps(func)
            def speed_wrapper(result, *args, **kwargs):
                in_value, out_value = func(result, *args, **kwargs)

                ms = result.speed[stage_id]
                offset_ms = sum(result.speed[s] for s in SPEED_STAGES[:index])
                start = getattr(result, "started_at", time.perf_counter()) + offset_ms / 1000

                recorder.log_pass(stage_id, ms, start, start + ms / 1000, describe(in_value), describe(out_value))

                return out_value

            return speed_wrapper

        @functools.wraps(func)
        def wall_wrapper(*args, **kwargs):
            recorder.set_active(stage_id)

            start = time.perf_counter()
            out_value = func(*args, **kwargs)
            end = time.perf_counter()

            in_desc = describe(args[0]) if args else "-"
            recorder.log_pass(stage_id, (end - start) * 1000, start, end, in_desc, describe(out_value))

            return out_value

        return wall_wrapper

    return decorator


def parse_docstring(func):
    """
    <summary>함수 docstring의 XML 태그를 읽어 딕셔너리로 만든다.</summary>
    <param name="func">docstring을 읽을 함수</param>
    <input>inspect.getdoc(func)로 얻은 docstring</input>
    <output>{"summary", "input", "output", "reason", "params": [{"name", "text"}]}</output>
    <reason>
    docstring은 루트 태그가 없는 XML 조각이라 그대로는 파싱되지 않는다.
    가짜 루트 태그로 감싼 뒤 ElementTree로 읽는다.
    여러 줄에 걸친 문장은 공백 하나로 이어 붙여 화면에서 자연스럽게 읽히게 한다.
    </reason>
    """
    doc = inspect.getdoc(func)

    if not doc:
        raise ValueError("docstring이 없습니다")

    root = ET.fromstring(f"<doc>{doc}</doc>")

    def text_of(element):
        if element is None:
            return ""

        return " ".join("".join(element.itertext()).split())

    return {
        "summary": text_of(root.find("summary")),
        "input": text_of(root.find("input")),
        "output": text_of(root.find("output")),
        "reason": text_of(root.find("reason")),
        "params": [{"name": p.get("name", ""), "text": text_of(p)} for p in root.findall("param")],
    }


def pipeline_metadata():
    """
    <summary>등록된 단계 목록과 docstring 설명을 플로우차트용 데이터로 만든다.</summary>
    <input>@stage로 등록된 단계 함수들</input>
    <output>[{id, label, summary, input, output, reason, params}] 리스트 (STAGE_ORDER 순서)</output>
    <reason>
    코드의 주석이 곧 화면 설명이 되므로, 설명을 고치려면 docstring만 고치면 된다.
    파싱에 실패한 단계도 플로우차트에서 빠지지 않도록 label만 채워 돌려주고 서버 로그에 경고를 남긴다.
    </reason>
    """
    items = []

    for stage_id in STAGE_ORDER:
        entry = _registry.get(stage_id)
        item = {"id": stage_id, "label": stage_id, "summary": "", "input": "", "output": "", "reason": "", "params": []}

        if entry is None:
            print(f"[경고] '{stage_id}' 단계에 등록된 함수가 없습니다.")
            items.append(item)
            continue

        item["label"] = entry["label"]

        try:
            item.update(parse_docstring(entry["func"]))
        except (ET.ParseError, ValueError) as error:
            print(f"[경고] '{stage_id}' 단계 docstring을 파싱하지 못했습니다: {error}")

        items.append(item)

    return items
