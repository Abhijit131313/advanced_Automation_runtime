# app.py
import os
import uuid
import json
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any, Tuple
from string import Template

import asyncio
import aiohttp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# FastAPI App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Workbench Studio Backend", version="0.1.0")

# Open CORS for frontend (Bolt). Keep credentials OFF on the client.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict to your Bolt origin
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # with '*' allow_origins, credentials must be False
)


# -----------------------------------------------------------------------------
# Environment / Config
# -----------------------------------------------------------------------------
GROK_API_URL = os.getenv("GROK_API_URL", "").strip()
GROK_API_KEY = os.getenv("GROK_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5")
DISABLE_LLM = os.getenv("DISABLE_LLM", "").strip() == "1"

# In-memory store (swap to DB for production)
STORE: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------
class SuggestedRuntime(BaseModel):
    key: str
    name: str
    confidence: float
    reason: str


class SopUploadResponse(BaseModel):
    sop_id: str
    summary: str
    automated_actions: List[str]
    manual_actions: List[str]
    suggested_runtimes: List[SuggestedRuntime]


class GenerateCodeRequest(BaseModel):
    sop_id: str
    runtime_key: str
    options: Optional[Dict[str, Any]] = {}


class CodeFile(BaseModel):
    path: str
    content: str


class GenerateCodeResponse(BaseModel):
    code_files: List[CodeFile]
    main_file: str
    editor_mode: str  # "xml" | "java" | "yaml" | "text"
    ui_schema: Optional[Dict[str, Any]]
    confidence: float


class RunTestRequest(BaseModel):
    sop_id: str
    runtime_key: str
    code: Optional[str] = None
    test_input: Optional[Dict[str, Any]] = {}


class RunTestResponse(BaseModel):
    trace: List[Dict[str, Any]]
    errors: List[str]
    metrics: Dict[str, Any]


class ChatRequest(BaseModel):
    sop_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    meta: Optional[Dict[str, Any]] = {}


class DiagramNode(BaseModel):
    id: str
    label: str
    type: Optional[str] = None


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None


class DiagramResponse(BaseModel):
    nodes: List[DiagramNode]
    edges: List[DiagramEdge]
    meta: Optional[Dict[str, Any]] = {}


class ChatEditRequest(BaseModel):
    sop_id: str
    message: str
    target_file: Optional[str] = None


class ChatEditSuggestion(BaseModel):
    path: str
    new_content: str
    explanation: Optional[str] = None


class ChatEditResponse(BaseModel):
    reply: str
    suggestion: Optional[ChatEditSuggestion] = None
    meta: Optional[Dict[str, Any]] = {}


class ApplyCodeEditRequest(BaseModel):
    sop_id: str
    path: str
    new_content: str
    author: Optional[str] = "agent"


class ApplyCodeEditResponse(BaseModel):
    success: bool
    code_files: List[CodeFile]
    version_id: str
    meta: Optional[Dict[str, Any]] = {}


# -----------------------------------------------------------------------------
# Helpers: Parsing, LLM, Runtime Suggestion, Summarization
# -----------------------------------------------------------------------------
def extract_steps_from_text(text: str) -> List[str]:
    """Heuristic step extraction from raw text."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    steps: List[str] = []
    for ln in lines:
        m1 = re.match(r"^\d+\.\s+(.*)", ln)
        if m1:
            steps.append(m1.group(1))
            continue
        m2 = re.match(r"^[-*]\s+(.*)", ln)
        if m2:
            steps.append(m2.group(1))
            continue
        if 6 < len(ln) < 300 and (ln.endswith(".") or ln.startswith("Step") or len(ln.split()) < 20):
            steps.append(ln)
    if not steps:
        paragraphs = [p.strip() for p in (text or "").split("\n\n") if p.strip()]
        steps = paragraphs[:20]
    return steps


async def call_ai_model(prompt: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    """Try Grok/xAI (OpenAI-style) safely; return 'LLM_NOT_CONFIGURED' when unusable."""
    if DISABLE_LLM or not (GROK_API_URL and GROK_API_KEY):
        await asyncio.sleep(0.01)
        return "LLM_NOT_CONFIGURED"

    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}

    def parse_ai_text(payload: Any) -> Optional[str]:
        if isinstance(payload, dict):
            if payload.get("error"):
                return None
            if "choices" in payload and payload["choices"]:
                ch = payload["choices"][0]
                if isinstance(ch, dict):
                    msg = ch.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]
                    if isinstance(ch.get("text"), str):
                        return ch["text"]
            if isinstance(payload.get("output"), str):
                return payload["output"]
        elif isinstance(payload, str) and payload.strip():
            low = payload.lower()
            if "bad data" in low or "messages cannot be empty" in low or "error" in low:
                return None
            return payload
        return None

    async with aiohttp.ClientSession() as session:
        # 1) Chat messages
        try:
            body = {
                "model": DEFAULT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            async with session.post(GROK_API_URL, headers=headers, json=body, timeout=60) as resp:
                txt = await resp.text()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = txt
                out = parse_ai_text(data)
                if out:
                    return out
        except Exception:
            pass

        # 2) Legacy 'prompt'
        try:
            body = {"model": DEFAULT_MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
            async with session.post(GROK_API_URL, headers=headers, json=body, timeout=60) as resp:
                txt = await resp.text()
                try:
                    data = json.loads(txt)
                except Exception:
                    data = txt
                out = parse_ai_text(data)
                if out:
                    return out
        except Exception:
            pass

    return "LLM_NOT_CONFIGURED"


def heuristic_suggest_runtimes(steps: List[str]) -> List[SuggestedRuntime]:
    """Basic scoring to propose top-3 runtimes."""
    text = " ".join(steps).lower()
    scores = {
        "bpmn": 0.5 + (0.1 if any(k in text for k in ["manual", "approval", "human"]) else 0.0),
        "camel": 0.5 + (0.2 if any(k in text for k in ["integration", "http", "api"]) else 0.0),
        "knative": 0.45 + (0.2 if any(k in text for k in ["event", "kafka", "queue"]) else 0.0),
        "temporal": 0.55 + (0.15 if any(k in text for k in ["retry", "long running", "saga"]) else 0.0),
    }
    mapping = {
        "bpmn": "BPMN (XML) - good for human tasks & manual approvals",
        "camel": "Apache Camel (Java DSL) - good for API/adapter integration",
        "knative": "Knative (YAML) - good for event-driven serverless",
        "temporal": "Temporal (Java/TS) - good for durable code-first workflows",
    }
    suggested = [
        SuggestedRuntime(key=k, name=k.upper(), confidence=round(min(0.99, v), 2), reason=mapping[k])
        for k, v in scores.items()
    ]
    return sorted(suggested, key=lambda s: s.confidence, reverse=True)[:3]


async def ai_suggest_runtimes(steps: List[str]) -> List[SuggestedRuntime]:
    """Ask LLM to pick top-3 runtimes; returns [] on failure."""
    prompt = (
        "You are an automation advisor. Given the following SOP steps, return a JSON array of the top 3 "
        "recommended automation runtimes. Each item MUST have keys: "
        "`key` (one of: bpmn, camel, knative, temporal), `name`, `confidence` (0.0-1.0), `reason` (short). "
        "Output VALID JSON array only (no extra commentary).\n\nSOP Steps:\n"
    )
    for i, s in enumerate(steps, start=1):
        prompt += f"{i}. {s}\n"
    prompt += (
        '\nExample: [{"key":"bpmn","name":"BPMN","confidence":0.92,"reason":"best for human approvals"},'
        '{"key":"camel","name":"Apache Camel","confidence":0.82,"reason":"API integrations"},'
        '{"key":"temporal","name":"Temporal","confidence":0.76,"reason":"durable code-first"}]\n'
    )

    ai_text = await call_ai_model(prompt, max_tokens=600, temperature=0.0)
    if not ai_text or ai_text.strip() == "LLM_NOT_CONFIGURED":
        return []

    start = ai_text.find("[")
    end = ai_text.rfind("]")
    parsed = None
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(ai_text[start : end + 1])
        except Exception:
            parsed = None
    if parsed is None:
        try:
            parsed = json.loads(ai_text)
        except Exception:
            parsed = None
    if not isinstance(parsed, list) or not parsed:
        return []

    out: List[SuggestedRuntime] = []
    allowed = {"bpmn", "camel", "knative", "temporal"}
    for item in parsed[:3]:
        if not isinstance(item, dict):
            continue
        k = item.get("key")
        if not isinstance(k, str):
            continue
        k = k.strip().lower()
        if k not in allowed:
            continue
        name = item.get("name") or k.upper()
        reason = item.get("reason", "")
        try:
            conf = float(item.get("confidence", 0))
        except Exception:
            conf = 0.0
        out.append(
            SuggestedRuntime(
                key=k,
                name=str(name),
                confidence=round(max(0.0, min(0.99, conf)), 2),
                reason=str(reason),
            )
        )
    return out


async def ai_summarize_sop(steps: List[str]) -> Tuple[Optional[str], Optional[List[str]], Optional[List[str]]]:
    """Ask LLM to split automated vs manual and summarize."""
    prompt = (
        "You are an automation analyst. Given the numbered SOP steps below, return a JSON object with keys:\n"
        "  summary (1-3 sentences), automated_actions (array of step texts), manual_actions (array of step texts).\n"
        "Output ONLY valid JSON (no commentary).\n\nSteps:\n"
    )
    for i, s in enumerate(steps, start=1):
        prompt += f"{i}. {s}\n"

    ai_text = await call_ai_model(prompt, max_tokens=800, temperature=0.0)
    if not ai_text or ai_text.strip() == "LLM_NOT_CONFIGURED":
        return None, None, None

    start = ai_text.find("{")
    end = ai_text.rfind("}")
    parsed = None
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(ai_text[start : end + 1])
        except Exception:
            parsed = None
    if parsed is None:
        try:
            parsed = json.loads(ai_text)
        except Exception:
            parsed = None
    if not isinstance(parsed, dict):
        return None, None, None

    summary = parsed.get("summary")
    automated = parsed.get("automated_actions") or parsed.get("automated") or []
    manual = parsed.get("manual_actions") or parsed.get("manual") or []
    if summary is None or not isinstance(automated, list) or not isinstance(manual, list):
        return None, None, None
    return str(summary), [str(x) for x in automated], [str(x) for x in manual]


def build_summary_and_action_split(steps: List[str]) -> Tuple[str, List[str], List[str]]:
    """Heuristic fallback split."""
    automated: List[str] = []
    manual: List[str] = []
    for s in steps:
        low = s.lower()
        if any(k in low for k in ["manual", "human", "approve", "sign", "call customer"]):
            manual.append(s)
        else:
            automated.append(s)
    summary = f"Detected {len(steps)} steps. Will automate {len(automated)}; {len(manual)} remain manual."
    return summary, automated, manual


# -----------------------------------------------------------------------------
# Code Generation Templates (BPMN/Camel/Knative/Temporal)
# -----------------------------------------------------------------------------
def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def escape_yaml(s: str) -> str:
    return s.replace("\n", " ").replace('"', "'")


def escape_java_comment(s: str) -> str:
    return s.replace("/*", "/ *").replace("*/", "* /").replace("\n", " ")


def generate_bpmn_xml(sop_title: str, steps: List[str]) -> str:
    process_id = f"proc_{uuid.uuid4().hex[:8]}"
    tasks_xml = ""
    for i, step in enumerate(steps, start=1):
        tasks_xml += f'    <userTask id="task_{i}" name="{escape_xml(step[:60])}" />\n'
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<definitions id="Definitions_{process_id}" xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">\n'
        f'  <process id="{process_id}" name="{escape_xml(sop_title)}" isExecutable="true">\n'
        f"{tasks_xml}  </process>\n"
        "</definitions>\n"
    )
    return xml


def generate_camel_java_dsl(sop_title: str, steps: List[str]) -> str:
    class_name = f"RouteBuilder_{uuid.uuid4().hex[:6]}"
    lower = class_name.lower()

    route_steps = ""
    for i, s in enumerate(steps, start=1):
        if any(k in s.lower() for k in ["http", "api", "post", "get"]):
            route_steps += f'            .to("http://external.api/endpoint") // step {i}: {escape_java_comment(s)}\n'
        else:
            route_steps += f'            .to("log:step{i}") // step {i}: {escape_java_comment(s)}\n'

    tpl = Template(
        """package com.example.routes;

import org.apache.camel.builder.RouteBuilder;
import org.springframework.stereotype.Component;

@Component
public class $CLASS extends RouteBuilder {
    @Override
    public void configure() throws Exception {
        from("direct:workbench_$LOWER")
$STEPS        ;
    }
}
"""
    )
    return tpl.substitute(CLASS=class_name, LOWER=lower, STEPS=route_steps)


def generate_knative_yaml(sop_title: str, steps: List[str]) -> str:
    name = f"workbench-{uuid.uuid4().hex[:6]}"
    yaml = (
        "apiVersion: serving.knative.dev/v1\n"
        "kind: Service\n"
        "metadata:\n"
        f"  name: {name}\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        "      - image: gcr.io/example/workbench-worker:latest\n"
        "        env:\n"
    )
    for i, s in enumerate(steps, start=1):
        yaml += f'        - name: STEP_{i}\n          value: "{escape_yaml(s[:120])}"\n'
    return yaml


def generate_temporal_java(sop_title: str, steps: List[str]) -> str:
    workflow_interface = f"IWorkflow{uuid.uuid4().hex[:6]}"
    workflow_impl = f"WorkflowImpl{uuid.uuid4().hex[:6]}"

    activities = ""
    activity_calls = ""
    for i, s in enumerate(steps, start=1):
        activities += f"    // Activity {i}: {escape_java_comment(s)}\n"
        activity_calls += f"        activities.executeStep{i}();\n"

    tpl = Template(
        """package com.example.temporal;

import io.temporal.workflow.Workflow;

public interface $IFACE {
    void execute();
}

/* Implementation */
public class $IMPL implements $IFACE {
    private final Activities activities = Workflow.newActivityStub(Activities.class);

    @Override
    public void execute() {
$CALLS    }
}

/* Activities interface (skeleton) */
interface Activities {
$ACTS}
"""
    )
    return tpl.substitute(
        IFACE=workflow_interface,
        IMPL=workflow_impl,
        CALLS=activity_calls,
        ACTS=activities,
    )


# -----------------------------------------------------------------------------
# Visualization helpers
# -----------------------------------------------------------------------------
def strip_xml_namespaces(xml_text: str) -> str:
    return re.sub(r'\sxmlns(:\w+)?="[^"]+"', "", xml_text, flags=re.MULTILINE)


def visualize_bpmn_from_xml(xml_str: str) -> DiagramResponse:
    nodes: List[DiagramNode] = []
    edges: List[DiagramEdge] = []
    try:
        cleaned = strip_xml_namespaces(xml_str)
        root = ET.fromstring(cleaned)
        user_tasks = root.findall(".//userTask")
        if not user_tasks:
            user_tasks = root.findall(".//task") + root.findall(".//serviceTask")
        for idx, ut in enumerate(user_tasks, start=1):
            uid = ut.get("id") or f"task_{idx}"
            name = ut.get("name") or uid
            nodes.append(DiagramNode(id=uid, label=name, type="userTask"))
        for i in range(len(nodes) - 1):
            edges.append(DiagramEdge(source=nodes[i].id, target=nodes[i + 1].id))
    except Exception:
        return DiagramResponse(nodes=[], edges=[], meta={"note": "bpmn_parse_failed"})
    return DiagramResponse(nodes=nodes, edges=edges, meta={"runtime": "bpmn"})


def visualize_camel_from_java(java_str: str) -> DiagramResponse:
    nodes: List[DiagramNode] = []
    edges: List[DiagramEdge] = []
    try:
        m = re.search(r'from\(\s*["\']([^"\']+)["\']\s*\)', java_str)
        src = m.group(1) if m else "direct:start"
        src_id = f"from::{src}"
        nodes.append(DiagramNode(id=src_id, label=src, type="endpoint"))
        tos = re.findall(r'\.to\(\s*["\']([^"\']+)["\']\s*\)', java_str)
        prev = src_id
        for i, t in enumerate(tos, start=1):
            tid = f"to::{i}::{t}"
            nodes.append(DiagramNode(id=tid, label=t, type="endpoint"))
            edges.append(DiagramEdge(source=prev, target=tid))
            prev = tid
    except Exception:
        return DiagramResponse(nodes=[], edges=[], meta={"note": "camel_parse_failed"})
    return DiagramResponse(nodes=nodes, edges=edges, meta={"runtime": "camel"})


def visualize_knative_from_yaml(yaml_str: str) -> DiagramResponse:
    nodes: List[DiagramNode] = []
    edges: List[DiagramEdge] = []
    try:
        matches = re.findall(r'(?m)^\s*- name:\s*STEP_(\d+)\s*\n\s*value:\s*["\']?([^"\']+)["\']?', yaml_str)
        prev = None
        for idx, val in sorted(matches, key=lambda x: int(x[0])):
            nid = f"step_{idx}"
            nodes.append(DiagramNode(id=nid, label=val.strip(), type="step"))
            if prev:
                edges.append(DiagramEdge(source=prev, target=nid))
            prev = nid
        if not nodes:
            m = re.search(r"image:\s*([^\n]+)", yaml_str)
            img = m.group(1).strip() if m else "workbench-worker"
            nodes.append(DiagramNode(id="service", label=img, type="service"))
    except Exception:
        return DiagramResponse(nodes=[], edges=[], meta={"note": "knative_parse_failed"})
    return DiagramResponse(nodes=nodes, edges=edges, meta={"runtime": "knative"})


def visualize_temporal_from_java(java_str: str) -> DiagramResponse:
    nodes: List[DiagramNode] = []
    edges: List[DiagramEdge] = []
    try:
        calls = re.findall(r"executeStep(\d+)\s*\(\s*\)", java_str)
        seen: List[str] = []
        prev = None
        for c in calls:
            if c in seen:
                continue
            seen.append(c)
            nid = f"activity_{c}"
            nodes.append(DiagramNode(id=nid, label=f"Activity {c}", type="activity"))
            if prev:
                edges.append(DiagramEdge(source=prev, target=nid))
            prev = nid
    except Exception:
        return DiagramResponse(nodes=[], edges=[], meta={"note": "temporal_parse_failed"})
    return DiagramResponse(nodes=nodes, edges=edges, meta={"runtime": "temporal"})


def visualize_generated_code(code_files: List[Dict[str, Any]], runtime_key: str) -> DiagramResponse:
    content = ""
    preferred = {
        "bpmn": ["process.bpmn", "process.xml", ".bpmn", ".xml"],
        "camel": ["RouteBuilder.java", ".java"],
        "knative": ["service.yaml", ".yaml", ".yml"],
        "temporal": ["workflow.java", ".java"],
    }
    rk = (runtime_key or "").lower()
    for cf in code_files:
        path = cf.get("path", "")
        cnt = cf.get("content", "") or ""
        if rk == "bpmn" and any(p in path for p in preferred["bpmn"]):
            content = cnt
            break
        if rk == "camel" and any(p in path for p in preferred["camel"]):
            content = cnt
            break
        if rk == "knative" and any(p in path for p in preferred["knative"]):
            content = cnt
            break
        if rk == "temporal" and any(p in path for p in preferred["temporal"]):
            content = cnt
            break
    if not content and code_files:
        content = code_files[0].get("content", "")

    if not content:
        return DiagramResponse(nodes=[], edges=[], meta={"note": "no_code_found"})

    if rk == "bpmn":
        return visualize_bpmn_from_xml(content)
    if rk == "camel":
        return visualize_camel_from_java(content)
    if rk == "knative":
        return visualize_knative_from_yaml(content)
    if rk == "temporal":
        return visualize_temporal_from_java(content)
    return DiagramResponse(nodes=[DiagramNode(id="code", label=content[:200])], edges=[], meta={"note": "unknown_runtime"})


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/sop/upload", response_model=SopUploadResponse)
async def upload_sop(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    scenario_id: Optional[str] = Form(None),
):
    if not file and not text:
        raise HTTPException(status_code=400, detail="file or text required")

    content = ""
    if file:
        raw = await file.read()
        # Try utf-8
        try:
            content = raw.decode("utf-8")
        except Exception:
            # Minimal placeholder for non-utf8/PDFs; you can add pdfminer later
            content_type = (file.content_type or "").lower()
            if "pdf" in content_type or (file.filename or "").lower().endswith(".pdf"):
                content = "<binary or non-utf8 document>"
            else:
                content = "<binary or non-utf8 document>"
    else:
        content = text or ""

    steps = extract_steps_from_text(content)

    # LLM-first summary; fallback heuristics
    summary: Optional[str] = None
    automated: Optional[List[str]] = None
    manual: Optional[List[str]] = None
    try:
        s_sum, s_auto, s_manual = await ai_summarize_sop(steps)
        if s_sum is not None and isinstance(s_auto, list) and isinstance(s_manual, list):
            summary, automated, manual = s_sum, s_auto, s_manual
    except Exception:
        summary, automated, manual = None, None, None
    if summary is None:
        summary, automated, manual = build_summary_and_action_split(steps)

    # LLM-first runtimes; fallback heuristics
    try:
        suggested = await ai_suggest_runtimes(steps)
    except Exception:
        suggested = []
    if not suggested:
        suggested = heuristic_suggest_runtimes(steps)

    sop_id = str(uuid.uuid4())
    STORE[sop_id] = {
        "sop_text": content,
        "steps": steps,
        "summary": summary,
        "automated": automated,
        "manual": manual,
        "suggested": [s.dict() for s in suggested],
        "scenario_id": scenario_id,
    }

    return SopUploadResponse(
        sop_id=sop_id,
        summary=summary,
        automated_actions=automated or [],
        manual_actions=manual or [],
        suggested_runtimes=suggested,
    )


@app.post("/api/generate_code", response_model=GenerateCodeResponse)
async def generate_code(req: GenerateCodeRequest):
    sop = STORE.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")
    steps = sop.get("steps", [])[:20]
    title = (sop.get("scenario_id") or "workbench_sop")[:60]
    runtime = (req.runtime_key or "").lower()

    code_files: List[CodeFile] = []
    editor_mode = "text"
    confidence = 0.7

    # Try LLM first
    prompt = f"Generate a skeleton automation for runtime={runtime} for SOP titled: {title}\n\nSteps:\n"
    for i, s in enumerate(steps, start=1):
        prompt += f"{i}. {s}\n"

    ai_response = await call_ai_model(prompt, max_tokens=1500, temperature=0.0)

    def looks_like_error(s: str) -> bool:
        low = (s or "").lower()
        return any(
            term in low
            for term in ["error", "invalid", "bad data", "messages cannot be empty", "not configured", "llm_not_configured"]
        )

    use_llm = bool(ai_response and isinstance(ai_response, str) and len(ai_response.strip()) > 80 and not looks_like_error(ai_response))

    if use_llm:
        path = "automation.txt"
        if runtime == "bpmn":
            path = "process.bpmn"
            editor_mode = "xml"
        elif runtime == "camel":
            path = "RouteBuilder.java"
            editor_mode = "java"
        elif runtime == "knative":
            path = "service.yaml"
            editor_mode = "yaml"
        elif runtime == "temporal":
            path = "workflow.java"
            editor_mode = "java"
        code_files.append(CodeFile(path=path, content=ai_response))
        confidence = 0.85
    else:
        # Fallback to safe templates
        if runtime == "bpmn":
            content = generate_bpmn_xml(title, steps)
            code_files.append(CodeFile(path="process.bpmn", content=content))
            editor_mode = "xml"
            confidence = 0.9
        elif runtime == "camel":
            content = generate_camel_java_dsl(title, steps)
            code_files.append(CodeFile(path="RouteBuilder.java", content=content))
            editor_mode = "java"
            confidence = 0.88
        elif runtime == "knative":
            content = generate_knative_yaml(title, steps)
            code_files.append(CodeFile(path="service.yaml", content=content))
            editor_mode = "yaml"
            confidence = 0.82
        elif runtime == "temporal":
            content = generate_temporal_java(title, steps)
            code_files.append(CodeFile(path="workflow.java", content=content))
            editor_mode = "java"
            confidence = 0.86
        else:
            code_files.append(CodeFile(path="automation.txt", content="// Unsupported runtime"))
            editor_mode = "text"
            confidence = 0.5

    # Simple UI schema inference
    ui_schema: Dict[str, Any] = {"title": title, "fields": []}
    joined = " ".join(steps).lower()
    if "amount" in joined:
        ui_schema["fields"].append({"name": "amount", "type": "number", "label": "Amount"})
    if "email" in joined:
        ui_schema["fields"].append({"name": "email", "type": "string", "format": "email", "label": "Email"})
    if "date" in joined or "due" in joined:
        ui_schema["fields"].append({"name": "date", "type": "string", "format": "date", "label": "Date"})
    if not ui_schema["fields"]:
        ui_schema["fields"].append({"name": "input1", "type": "string", "label": "Input 1"})

    STORE[req.sop_id]["last_generated"] = {
        "runtime": runtime,
        "code_files": [cf.dict() for cf in code_files],
        "editor_mode": editor_mode,
        "ui_schema": ui_schema,
        "confidence": confidence,
        "version_id": uuid.uuid4().hex[:8],
    }

    return GenerateCodeResponse(
        code_files=code_files,
        main_file=code_files[0].path,
        editor_mode=editor_mode,
        ui_schema=ui_schema,
        confidence=confidence,
    )


@app.post("/api/run_test", response_model=RunTestResponse)
async def run_test(req: RunTestRequest):
    data = STORE.get(req.sop_id)
    if not data:
        raise HTTPException(status_code=404, detail="sop_id not found")

    steps = data.get("steps", [])
    trace: List[Dict[str, Any]] = []
    errors: List[str] = []

    for i, s in enumerate(steps[:10], start=1):
        trace.append({"step": i, "action": s, "status": "ok"})

    # Minimal linting for the provided code (if any)
    if req.code and "error" in req.code.lower():
        errors.append("Code contains the word 'error' (simulated lint check).")

    metrics = {"steps_executed": len(trace), "errors": len(errors)}
    return RunTestResponse(trace=trace, errors=errors, metrics=metrics)


@app.post("/api/chat_agent", response_model=ChatResponse)
async def chat_agent(req: ChatRequest):
    """Agentic helper chat; uses LLM if configured otherwise echoes back guidance."""
    data = STORE.get(req.sop_id)
    steps = data.get("steps", []) if data else []
    context = "SOP Steps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    if not DISABLE_LLM and GROK_API_URL and GROK_API_KEY:
        user = f"You are a coding assistant for automation runtimes. Context:\n{context}\n\nUser: {req.message}"
        out = await call_ai_model(user, max_tokens=800, temperature=0.2)
        if out and out.strip() != "LLM_NOT_CONFIGURED":
            return ChatResponse(reply=out.strip(), meta={"llm": True})

    # Fallback
    return ChatResponse(
        reply=(
            "LLM not available. Here's a tip: ensure each step is atomic and clearly states inputs/outputs. "
            "Ask me to generate code for a specific runtime or to refactor a step."
        ),
        meta={"llm": False},
    )


@app.post("/api/chat_agent_suggest_edit", response_model=ChatEditResponse)
async def chat_agent_suggest_edit(req: ChatEditRequest):
    data = STORE.get(req.sop_id)
    if not data or "last_generated" not in data:
        raise HTTPException(status_code=404, detail="no generated code for this sop_id")

    last = data["last_generated"]
    code_files = last.get("code_files", [])
    target_path = req.target_file or (code_files[0]["path"] if code_files else None)
    if not target_path:
        raise HTTPException(status_code=400, detail="no code files present to edit")

    current = next((cf for cf in code_files if cf["path"] == target_path), None)
    if not current:
        raise HTTPException(status_code=404, detail="target_file not found")

    base_text = current["content"]
    prompt = (
        "You are a senior automation engineer. Given the user's instruction, propose a full-file replacement edit.\n"
        "Output ONLY the new file content (no commentary).\n\n"
        f"User instruction:\n{req.message}\n\nCurrent file ({target_path}):\n{base_text}\n"
    )

    if not DISABLE_LLM and GROK_API_URL and GROK_API_KEY:
        out = await call_ai_model(prompt, max_tokens=1600, temperature=0.2)
        if out and out.strip() != "LLM_NOT_CONFIGURED":
            suggestion = ChatEditSuggestion(path=target_path, new_content=out, explanation="Proposed edit from agent.")
            return ChatEditResponse(reply="Proposed edit generated.", suggestion=suggestion, meta={"llm": True})

    # Fallback: append a comment
    new_content = base_text + "\n// NOTE: Agent suggests reviewing edge cases and adding validations.\n"
    suggestion = ChatEditSuggestion(path=target_path, new_content=new_content, explanation="Fallback edit suggestion.")
    return ChatEditResponse(reply="LLM unavailable; provided fallback suggestion.", suggestion=suggestion, meta={"llm": False})


@app.post("/api/apply_code_edit", response_model=ApplyCodeEditResponse)
async def apply_code_edit(req: ApplyCodeEditRequest):
    data = STORE.get(req.sop_id)
    if not data or "last_generated" not in data:
        raise HTTPException(status_code=404, detail="no generated code for this sop_id")

    last = data["last_generated"]
    code_files = last.get("code_files", [])
    updated = False
    for cf in code_files:
        if cf["path"] == req.path:
            cf["content"] = req.new_content
            updated = True
            break

    if not updated:
        # If file not found, add as a new file
        code_files.append({"path": req.path, "content": req.new_content})

    version_id = uuid.uuid4().hex[:8]
    last["version_id"] = version_id
    last["code_files"] = code_files
    return ApplyCodeEditResponse(
        success=True,
        code_files=[CodeFile(**cf) for cf in code_files],
        version_id=version_id,
        meta={"author": req.author},
    )


@app.post("/api/visualize_code", response_model=DiagramResponse)
async def visualize_code_endpoint(payload: Dict[str, Any]):
    sop_id = payload.get("sop_id")
    runtime_key = (payload.get("runtime_key") or "").lower()
    if not sop_id or sop_id not in STORE:
        raise HTTPException(status_code=404, detail="sop_id not found")

    last = STORE[sop_id].get("last_generated")
    if not last:
        return DiagramResponse(nodes=[], edges=[], meta={"note": "no_generated_code"})

    code_files = last.get("code_files", [])
    return visualize_generated_code(code_files, runtime_key)


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": os.getenv("RENDER_INSTANCE_START_TIME", "n/a")}
