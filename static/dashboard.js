/*
 * STEP 8 대시보드 스크립트
 *
 * 1. /api/pipeline 으로 단계 목록을 받아 플로우차트(인라인 SVG)를 만든다.
 * 2. /events (SSE) 로 0.2초마다 집계 데이터를 받아 상단 바, 노드, 로그를 갱신한다.
 * 3. 촬영 버튼은 STEP 3와 같이 POST /capture 를 호출한다.
 *
 * 플로우차트를 Mermaid.js 대신 인라인 SVG로 그리는 이유:
 * - 0.2초마다 "활성 노드 강조, 노드별 ms, 흐림"만 바꾸면 된다.
 *   SVG는 해당 요소의 class와 글자만 바꾸면 되지만, Mermaid는 그림 전체를 다시 렌더링해야 한다.
 * - 노드 클릭, CSS 변수(다크/라이트) 적용이 일반 DOM과 똑같이 동작한다.
 * - 외부 CDN이 필요 없어 인터넷이 막힌 실습실에서도 동작한다.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

// 플로우차트 크기 (SVG 좌표 단위)
const CHART = {
    width: 280,
    nodeX: 30,
    nodeWidth: 200,
    nodeHeight: 44,
    gap: 34,
    top: 8,
};

// 로그 콘솔에 남겨 둘 최대 줄 수
const MAX_LOG_LINES = 400;

const state = {
    stages: [],
    nodes: new Map(),
    selectedId: null,
    paused: false,
};

const elements = {
    flowchart: document.getElementById("flowchart"),
    bottleneck: document.getElementById("bottleneck"),
    log: document.getElementById("log"),
    pauseButton: document.getElementById("pause-button"),
    clearButton: document.getElementById("clear-button"),
    fps: document.getElementById("stat-fps"),
    frame: document.getElementById("stat-frame"),
    status: document.getElementById("stat-status"),
    detailTitle: document.getElementById("detail-title"),
    detailHint: document.getElementById("detail-hint"),
    detailBody: document.getElementById("detail-body"),
    detailSummary: document.getElementById("detail-summary"),
    detailInput: document.getElementById("detail-input"),
    detailOutput: document.getElementById("detail-output"),
    detailReason: document.getElementById("detail-reason"),
    detailParams: document.getElementById("detail-params"),
    captureButton: document.getElementById("capture-button"),
    captureStatus: document.getElementById("capture-status"),
    history: document.getElementById("history"),
    historyEmpty: document.getElementById("history-empty"),
};


// SVG 요소를 만들고 속성을 한 번에 지정한다.
function svg(tag, attributes = {}) {
    const element = document.createElementNS(SVG_NS, tag);

    for (const [key, value] of Object.entries(attributes)) {
        element.setAttribute(key, value);
    }

    return element;
}


// /api/pipeline 데이터로 위에서 아래로 흐르는(TD) 플로우차트를 만든다.
function buildFlowchart(stages) {
    const count = stages.length;
    const height = CHART.top * 2 + count * CHART.nodeHeight + (count - 1) * CHART.gap;
    const root = svg("svg", { viewBox: `0 0 ${CHART.width} ${height}`, role: "img", "aria-label": "파이프라인 단계 흐름" });

    // 화살촉 정의
    const defs = svg("defs");
    const marker = svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" });
    marker.append(svg("path", { d: "M0,0 L10,5 L0,10 z", class: "arrowhead" }));
    defs.append(marker);
    root.append(defs);

    const centerX = CHART.nodeX + CHART.nodeWidth / 2;

    stages.forEach((stage, index) => {
        const y = CHART.top + index * (CHART.nodeHeight + CHART.gap);

        // 노드 사이 화살표와, 그 옆에 평균 ms 글자
        if (index < count - 1) {
            const edgeTop = y + CHART.nodeHeight;
            const edgeBottom = edgeTop + CHART.gap;

            root.append(svg("line", { x1: centerX, y1: edgeTop, x2: centerX, y2: edgeBottom - 1, class: "edge", "marker-end": "url(#arrow)" }));
        }

        const group = svg("g", { class: "node", tabindex: 0, role: "listitem", "data-id": stage.id });
        group.append(svg("rect", { x: CHART.nodeX, y, width: CHART.nodeWidth, height: CHART.nodeHeight, rx: 1 }));

        const idText = svg("text", { x: CHART.nodeX + 12, y: y + 19, class: "node-id" });
        idText.textContent = stage.id;

        const labelText = svg("text", { x: CHART.nodeX + 12, y: y + 35, class: "node-label" });
        labelText.textContent = stage.label;

        const flagText = svg("text", { x: CHART.nodeX + CHART.nodeWidth - 10, y: y + 19, class: "node-flag", "text-anchor": "end" });
        flagText.textContent = "병목";

        // 노드 아래(화살표 오른쪽)에 평균 소요 ms
        const msText = svg("text", { x: centerX + 10, y: y + CHART.nodeHeight + 21, class: "node-ms" });
        msText.textContent = "- ms";

        const title = svg("title");
        title.textContent = stage.summary || stage.label;

        group.append(idText, labelText, flagText, title);
        root.append(group, msText);

        group.addEventListener("click", () => showDetail(stage.id));
        group.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                showDetail(stage.id);
            }
        });

        state.nodes.set(stage.id, { group, msText });
    });

    elements.flowchart.replaceChildren(root);
}


// 노드를 클릭하면 docstring에서 읽은 설명을 상세 패널에 보여준다.
function showDetail(stageId) {
    const stage = state.stages.find((item) => item.id === stageId);

    if (!stage) {
        return;
    }

    state.selectedId = stageId;

    for (const [id, node] of state.nodes) {
        node.group.classList.toggle("selected", id === stageId);
    }

    elements.detailTitle.textContent = `${stage.id} · ${stage.label}`;
    elements.detailHint.hidden = true;
    elements.detailBody.hidden = false;

    const empty = "(docstring 정보 없음)";
    elements.detailSummary.textContent = stage.summary || empty;
    elements.detailInput.textContent = stage.input || empty;
    elements.detailOutput.textContent = stage.output || empty;
    elements.detailReason.textContent = stage.reason || empty;

    elements.detailParams.replaceChildren();

    if (stage.params.length === 0) {
        const item = document.createElement("li");
        item.textContent = "없음";
        elements.detailParams.append(item);
    }

    for (const param of stage.params) {
        const item = document.createElement("li");
        const name = document.createElement("code");
        name.textContent = param.name;
        item.append(name, ` — ${param.text}`);
        elements.detailParams.append(item);
    }
}


// SSE 데이터로 노드 강조, 흐림, ms, 병목을 갱신한다.
function updateFlowchart(data) {
    const total = data.total_ms;

    for (const [id, node] of state.nodes) {
        const ms = data.stages[id];

        node.group.classList.toggle("active", data.active === id);
        node.group.classList.toggle("bottleneck", data.bottleneck === id);
        node.group.classList.toggle("dimmed", id === "inference" && data.skipped);

        if (ms === undefined) {
            node.msText.textContent = "- ms";
        } else {
            const share = total > 0 ? (ms / total) * 100 : 0;
            node.msText.textContent = `${ms.toFixed(1)} ms · ${share.toFixed(0)}%`;
        }
    }

    if (data.bottleneck) {
        const ms = data.stages[data.bottleneck];
        const share = total > 0 ? (ms / total) * 100 : 0;

        elements.bottleneck.replaceChildren("병목: ");
        const strong = document.createElement("strong");
        strong.textContent = data.bottleneck;
        elements.bottleneck.append(strong, ` ${ms.toFixed(1)} ms (${share.toFixed(0)}%) / 합계 ${total.toFixed(1)} ms`);
    }
}


// 상단 바의 FPS, 프레임 번호, 상태를 갱신한다.
function updateTopbar(data) {
    elements.fps.textContent = data.fps.toFixed(1);
    elements.frame.textContent = String(data.frame).padStart(4, "0");
    elements.status.textContent = data.running ? "실행 중" : "중지됨";
}


// 새 로그 줄을 아래에 붙이고 자동 스크롤한다. 일시정지 중에는 버린다.
function appendLogs(logs) {
    if (state.paused || logs.length === 0) {
        return;
    }

    const fragment = document.createDocumentFragment();

    for (const log of logs) {
        const line = document.createElement("span");
        line.className = `log-line level-${log.level}`;
        line.textContent = log.text;
        fragment.append(line);
    }

    elements.log.append(fragment);

    while (elements.log.childElementCount > MAX_LOG_LINES) {
        elements.log.firstElementChild.remove();
    }

    elements.log.scrollTop = elements.log.scrollHeight;
}


// SSE 연결: 서버가 0.2초마다 보내는 집계 데이터를 받는다. 끊기면 브라우저가 자동 재연결한다.
function connectEvents() {
    const source = new EventSource("/events");

    source.onmessage = (event) => {
        const data = JSON.parse(event.data);

        updateTopbar(data);
        updateFlowchart(data);
        appendLogs(data.logs);
    };

    source.onerror = () => {
        elements.status.textContent = "연결 끊김 · 재시도 중";
    };
}


// 로그 일시정지 버튼과 레벨 필터
function setupConsole() {
    elements.pauseButton.addEventListener("click", () => {
        state.paused = !state.paused;
        elements.pauseButton.setAttribute("aria-pressed", String(state.paused));
        elements.pauseButton.textContent = state.paused ? "재개" : "일시정지";
    });

    elements.clearButton.addEventListener("click", () => {
        elements.log.replaceChildren();
    });

    for (const checkbox of document.querySelectorAll(".console-toolbar input[data-level]")) {
        checkbox.addEventListener("change", () => {
            elements.log.classList.toggle(`hide-${checkbox.dataset.level}`, !checkbox.checked);
            elements.log.scrollTop = elements.log.scrollHeight;
        });
    }
}


// 촬영 이력 맨 앞에 썸네일을 추가하고 최근 N장만 남긴다.
function addThumbnail(filename) {
    const url = "/captures/" + encodeURIComponent(filename);
    const item = document.createElement("li");
    const link = document.createElement("a");
    const image = document.createElement("img");
    const label = document.createElement("span");

    link.href = url;
    link.target = "_blank";
    link.rel = "noopener";
    image.src = url;
    image.alt = filename;
    label.textContent = filename;

    link.append(image, label);
    item.append(link);
    elements.history.prepend(item);

    while (elements.history.children.length > window.DASHBOARD.historyLimit) {
        elements.history.lastElementChild.remove();
    }

    elements.historyEmpty.hidden = true;
}


// 촬영 버튼: POST /capture → 탐지 결과 프레임 저장 (STEP 3 기능 유지)
function setupCapture() {
    elements.captureButton.addEventListener("click", async () => {
        elements.captureButton.disabled = true;
        elements.captureStatus.textContent = "저장 중...";

        try {
            const response = await fetch("/capture", { method: "POST" });
            const data = await response.json();

            if (!response.ok) {
                elements.captureStatus.textContent = data.error || "촬영에 실패했습니다.";
                return;
            }

            addThumbnail(data.filename);
            elements.captureStatus.textContent = "저장됨: " + data.filename;
        } catch (error) {
            elements.captureStatus.textContent = "서버에 연결할 수 없습니다.";
        } finally {
            elements.captureButton.disabled = false;
        }
    });
}


// 시작: 플로우차트를 먼저 만든 뒤 SSE를 연결한다.
async function start() {
    setupConsole();
    setupCapture();

    try {
        const response = await fetch("/api/pipeline");
        state.stages = await response.json();
        buildFlowchart(state.stages);
    } catch (error) {
        elements.flowchart.textContent = "단계 정보를 불러오지 못했습니다.";
    }

    connectEvents();
}

start();
