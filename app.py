# app.py
import os
import io
import json
import uuid
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# -----------------------------
# OpenAI SDK (for Grok via base_url)
# -----------------------------
try:
    from openai import OpenAI  # openai==1.x
except Exception:  # pragma: no cover
    OpenAI = None  # fallback handled later


# -----------------------------
# App setup & logging
# -----------------------------
logger = logging.getLogger("workbench-backend")
logging.basicConfig(level=logging.INFO, format="INFO:workbench-backend:%(message)s")

app = FastAPI(title="Workbench Studio Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# In-memory stores (demo)
# -----------------------------
SOP_STORE: Dict[str, Dict[str, Any]] = {}
CODE_STORE: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# Environment / LLM client
# -----------------------------
GROK_BASE_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1").rstrip("/")
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5")

_proxy = (
    os.getenv("OUTBOUND_HTTP_PROXY")
    or os.getenv("HTTPS_PROXY")
    or os.getenv("HTTP_PROXY")
)

_http_client = None
if _proxy:
    try:
        _http_client = httpx.Client(proxies=_proxy, timeout=30.0)
    except Exception as e:  # pragma: no cover
        logger.info(f"httpx client proxy setup failed, proceeding without proxy: {e}")
        _http_client = None

client = None
if OpenAI and GROK_API_KEY:
    try:
        client = OpenAI(
            base_url=GROK_BASE_URL,
            api_key=GROK_API_KEY,
            http_client=_http_client,
        )
        MODEL = DEFAULT_MODEL
    except TypeError as e:
        # Safety: avoid the 'proxies' kwarg crash entirely
        logger.info(f"OpenAI init error (will fallback heuristic): {e}")
        client = None
        MODEL = None
else:
    MODEL = None


# -----------------------------
# Helpers
# -----------------------------
def read_text_from_upload(file: UploadFile) -> str:
    name = (file.filename or "").lower()
    raw = file.file.read()
    # very-simple: if it's a PDF, we won't parse in this demo; just mark placeholder
    if name.endswith(".pdf"):
        return "[[PDF uploaded; using filename as title and placeholder steps]]"
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_title_and_steps(text: str, scenario_id: Optional[str]) -> (str, List[str]):
    """Best-effort step extraction from plain text."""
    title = (scenario_id or "scenario") + "-" + str(uuid.uuid1().int % 10_000_000_000)
    lines = [l.strip() for l in text.splitlines()]
    # extract non-empty bullets/numbered lines or fall back to any non-empty
    steps: List[str] = []
    for l in lines:
        if not l:
            continue
        if l[:2] in ("- ", "* "):
            steps.append(l[2:].strip())
        elif l[:3].isdigit() or (len(l) > 2 and l[0].isdigit() and l[1] == "."):
            # "1." style
            steps.append(l.split(".", 1)[-1].strip())
        else:
            # Heuristic: short lines -> section, long lines -> potential step if imperative
            if len(l) > 10:
                steps.append(l)
    if not steps:
        steps = ["Start", "Do work", "Finish"]
    return title, steps


def suggest_runtimes(steps: List[str]) -> List[Dict[str, Any]]:
    """Return top-3 runtime suggestions with confidence & reason."""
    # trivial heuristic for demo
    text = " ".join(steps).lower()
    candidates = []
    candidates.append({
        "key": "temporal",
        "name": "TEMPORAL",
        "confidence": 0.55 if "retry" in text or "orchestrate" in text else 0.5,
        "reason": "Temporal (Java/TS) - durable, code-first workflows",
    })
    candidates.append({
        "key": "bpmn",
        "name": "BPMN",
        "confidence": 0.52 if any(x in text for x in ["approve", "review", "human"]) else 0.5,
        "reason": "BPMN (XML) - strong for human tasks & approvals",
    })
    candidates.append({
        "key": "camel",
        "name": "CAMEL",
        "confidence": 0.5,
        "reason": "Apache Camel (Java DSL) - integration & routing",
    })
    # ensure sorted desc
    candidates = sorted(candidates, key=lambda c: c["confidence"], reverse=True)[:3]
    return candidates


def classify_manual_vs_auto(steps: List[str]) -> (List[str], List[str]):
    """Very naive split: steps with 'review/approve' -> manual, else automated."""
    manual_keywords = ("approve", "approval", "review", "verify identity", "manual")
    manual, automated = [], []
    for s in steps:
        if any(k in s.lower() for k in manual_keywords):
            manual.append(s)
        else:
            automated.append(s)
    return automated, manual


# -----------------------------
# Code Generators (escape braces!)
# -----------------------------
def generate_bpmn_xml(title: str, steps: List[str]) -> str:
    # XML with escaped braces not needed; keep simple BPMN-ish diagram
    nodes_xml = []
    edges_xml = []
    prev_id = None
    for i, step in enumerate(steps, start=1):
        node_id = f"task_{i}"
        nodes_xml.append(
            f'    <bpmn:task id="{node_id}" name="{step.replace("&", "&amp;")}"/>\n'
        )
        if prev_id:
            edges_xml.append(
                f'    <bpmn:sequenceFlow id="flow_{i-1}_{i}" sourceRef="{prev_id}" targetRef="{node_id}"/>\n'
            )
        prev_id = node_id

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">\n'
        f'  <bpmn:process id="{title}" name="{title}">\n'
        + "".join(nodes_xml)
        + "".join(edges_xml)
        + "  </bpmn:process>\n"
        "</bpmn:definitions>\n"
    )
    return xml


def generate_camel_java_dsl(title: str, steps: List[str]) -> str:
    class_name = "".join([c for c in title.title() if c.isalnum()])
    route_steps = []
    for i, s in enumerate(steps, start=1):
        # keep simple 'log' steps; escape quotes
        msg = s.replace('"', '\\"')
        route_steps.append(f'            .process(exchange -> {{ /* step {i}: {msg} */ }})\n')
        route_steps.append(f'            .log("Step {i}: {msg}")\n')
    route_chain = "".join(route_steps).rstrip()

    # Use f-string and double-curly to escape Java braces where needed.
    template = (
        "package com.example.camel;\n"
        "import org.apache.camel.builder.RouteBuilder;\n\n"
        f"public class {class_name}RouteBuilder extends RouteBuilder {{\n"
        "    @Override\n"
        "    public void configure() throws Exception {\n"
        '        from("direct:start")\n'
        f"{route_chain};\n"
        "    }\n"
        "}\n"
    )
    return template


def generate_temporal_java(title: str, steps: List[str]) -> str:
    class_seed = "".join([c for c in title if c.isalnum()])[-6:] or "Flow"
    workflow_iface = f"IWorkflow{class_seed}"
    workflow_impl = f"WorkflowImpl{class_seed}"
    activities_iface = "Activities"

    # Build calls to activities.executeStepN();
    activity_calls = []
    comments = []
    for i, s in enumerate(steps, start=1):
        safe = s.replace("*/", "*\\/")  # prevent comment break
        activity_calls.append(f"        activities.executeStep{i}();\n")
        comments.append(f"    // Activity {i}: {safe}\n")

    activities_methods = []
    for i, s in enumerate(steps, start=1):
        safe = s.replace("*/", "*\\/")
        activities_methods.append(f"    void executeStep{i}(); {f'// {safe}'}\n")

    # Use f-strings; Java braces do not conflict here.
    code = (
        "package com.example.temporal;\n"
        "import io.temporal.workflow.Workflow;\n\n"
        f"public interface {workflow_iface} {{\n"
        "    void execute();\n"
        "}\n\n"
        f"public class {workflow_impl} implements {workflow_iface} {{\n"
        f"    private final {activities_iface} activities = Workflow.newActivityStub({activities_iface}.class);\n"
        "    @Override\n"
        "    public void execute() {\n"
        + "".join(activity_calls)
        + "    }\n"
        "}\n\n"
        f"interface {activities_iface} {{\n"
        + "".join(comments)
        + "\n"
        + "".join(activities_methods)
        + "}\n"
    )
    return code


def generate_knative_yaml(title: str, steps: List[str]) -> str:
    # Very basic Knative Service with annotations listing steps
    annotations = "\\n".join([f"- {s.replace(':', ' -')}" for s in steps])
    yaml = (
        "apiVersion: serving.knative.dev/v1\n"
        "kind: Service\n"
        f"metadata:\n  name: {title.lower().replace('_','-')[:50]}\n"
        "  annotations:\n"
        f"    workbench/steps: |\n      {annotations}\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        "        - image: gcr.io/knative-sample/helloworld-go\n"
        "          env:\n"
        "            - name: TARGET\n"
        f"              value: \"{title}\"\n"
    )
    return yaml


def pick_editor_mode(runtime_key: str) -> str:
    if runtime_key == "bpmn":
        return "xml"
    if runtime_key in ("camel", "temporal"):
        return "java"
    if runtime_key == "knative":
        return "yaml"
    return "plaintext"


def explain_code_with_llm(runtime_key: str, code: str, steps: List[str]) -> str:
    """Use Grok via OpenAI SDK if available, else heuristic."""
    system = (
        "You are a senior automation engineer. Explain the workflow automation code clearly and concisely "
        "for a process designer. Use bullets and short paragraphs. Avoid code unless necessary."
    )
    user = (
        f"Runtime: {runtime_key}\n\n"
        "Source Code:\n"
        "```\n"
        f"{code}\n"
        "```\n\n"
        "Also summarize in plain English what this automation does step by step."
    )
    if client and MODEL:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            out = resp.choices[0].message.content.strip()
            return out
        except Exception as e:
            logger.info(f"explain_code_text: LLM error, using heuristic. err={e}")

    # Heuristic fallback
    lines = [f"This automation is implemented for runtime **{runtime_key.upper()}**.",
             "It orchestrates the following steps in order:"]
    for i, s in enumerate(steps, start=1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)


# -----------------------------
# Pydantic Models
# -----------------------------
class UploadResponse(BaseModel):
    sop_id: str
    summary: str
    automated_actions: List[str]
    manual_actions: List[str]
    suggested_runtimes: List[Dict[str, Any]]


class GenerateCodeRequest(BaseModel):
    sop_id: str
    runtime_key: str
    options: Optional[Dict[str, Any]] = None


class CodeFile(BaseModel):
    path: str
    content: str


class GenerateCodeResponse(BaseModel):
    code_files: List[CodeFile]
    main_file: str
    editor_mode: str
    ui_schema: Dict[str, Any]
    confidence: float
    explanation: Optional[str] = None  # "What this automation code is doing?"


class RunTestRequest(BaseModel):
    sop_id: str


class RunTestResponse(BaseModel):
    success: bool
    logs: List[str]


class VisualizeRequest(BaseModel):
    sop_id: str
    runtime_key: Optional[str] = None


class VisualizeResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class ChatAgentRequest(BaseModel):
    sop_id: Optional[str] = None
    message: str


class ChatAgentResponse(BaseModel):
    reply: str


class SuggestEditRequest(BaseModel):
    sop_id: str
    instruction: str


class SuggestEditResponse(BaseModel):
    suggestion: str
    patch: Dict[str, Any]


class ApplyEditRequest(BaseModel):
    sop_id: str
    patch: Dict[str, Any]


class ApplyEditResponse(BaseModel):
    code_files: List[CodeFile]
    main_file: str


class ExplainRequest(BaseModel):
    runtime_key: str
    code: str
    steps: List[str]


class ExplainResponse(BaseModel):
    explanation: str


# -----------------------------
# Routes
# -----------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/sop/upload", response_model=UploadResponse)
async def upload_sop(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    scenario_id: Optional[str] = Form(None),
):
    if not file and not text:
        raise HTTPException(status_code=400, detail="Provide a file or text.")
    provided_text = text or ""
    if file:
        provided_text = read_text_from_upload(file) or provided_text

    title, steps = extract_title_and_steps(provided_text, scenario_id)
    automated, manual = classify_manual_vs_auto(steps)
    summary = f"Detected {len(steps)} steps. Will automate {len(automated)}; {len(manual)} remain manual."

    sop_id = str(uuid.uuid4())
    SOP_STORE[sop_id] = {
        "title": title,
        "steps": steps,
        "automated": automated,
        "manual": manual,
        "suggested_runtimes": suggest_runtimes(steps),
        "raw_text": provided_text,
    }
    logger.info(f"Uploaded SOP {sop_id} title={title} steps={len(steps)}")
    return {
        "sop_id": sop_id,
        "summary": summary,
        "automated_actions": automated if automated else ["<binary or non-utf8 document>"] if file and provided_text.startswith("[[PDF") else [],
        "manual_actions": manual,
        "suggested_runtimes": SOP_STORE[sop_id]["suggested_runtimes"],
    }


@app.post("/api/generate_code", response_model=GenerateCodeResponse)
async def generate_code(req: GenerateCodeRequest):
    sop = SOP_STORE.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")

    runtime_key = req.runtime_key.lower()
    steps = sop["steps"]
    title = sop["title"]

    if runtime_key == "bpmn":
        content = generate_bpmn_xml(title, steps)
        main_file = f"{title}.bpmn.xml"
    elif runtime_key == "camel":
        content = generate_camel_java_dsl(title, steps)
        main_file = f"{title}RouteBuilder.java"
    elif runtime_key == "temporal":
        content = generate_temporal_java(title, steps)
        main_file = f"{title}_TemporalWorkflow.java"
    elif runtime_key == "knative":
        content = generate_knative_yaml(title, steps)
        main_file = f"{title}.knative.yaml"
    else:
        raise HTTPException(status_code=400, detail="Unsupported runtime_key")

    editor_mode = pick_editor_mode(runtime_key)
    ui_schema = {"runtime": runtime_key, "title": title, "stepCount": len(steps)}

    # confidence from suggested runtimes
    confidence = 0.5
    for r in sop["suggested_runtimes"]:
        if r["key"] == runtime_key:
            confidence = r["confidence"]
            break

    # Generate explanation (Grok if available; else heuristic)
    explanation = explain_code_with_llm(runtime_key, content, steps)

    code_files = [{"path": main_file, "content": content}]
    CODE_STORE[req.sop_id] = {
        "runtime": runtime_key,
        "code_files": code_files,
        "main_file": main_file,
        "editor_mode": editor_mode,
        "ui_schema": ui_schema,
        "confidence": confidence,
        "explanation": explanation,
        "steps": steps,
        "title": title,
    }

    logger.info(
        f"Generated code for sop_id={req.sop_id} runtime={runtime_key} (explain: {'LLM' if client else 'heuristic'})"
    )

    return {
        "code_files": code_files,
        "main_file": main_file,
        "editor_mode": editor_mode,
        "ui_schema": ui_schema,
        "confidence": confidence,
        "explanation": explanation,
    }


@app.post("/api/run_test", response_model=RunTestResponse)
async def run_test(req: RunTestRequest):
    code = CODE_STORE.get(req.sop_id)
    if not code:
        raise HTTPException(status_code=404, detail="sop_id not found or code not generated")
    logs = [
        "Compiling/validating generated automation...",
        f"Runtime detected: {code['runtime']}",
        f"Main file: {code['main_file']}",
        "All checks passed (simulated).",
    ]
    return {"success": True, "logs": logs}


@app.post("/api/visualize_code", response_model=VisualizeResponse)
async def visualize_code(req: VisualizeRequest):
    sop = SOP_STORE.get(req.sop_id)
    code = CODE_STORE.get(req.sop_id)
    if not sop or not code:
        raise HTTPException(status_code=404, detail="sop_id not found or code not generated")

    steps = sop["steps"]
    nodes = [{"id": f"step_{i}", "label": s} for i, s in enumerate(steps, start=1)]
    edges = [{"from": f"step_{i}", "to": f"step_{i+1}"} for i in range(1, len(steps))]
    return {"nodes": nodes, "edges": edges}


@app.post("/api/chat_agent", response_model=ChatAgentResponse)
async def chat_agent(req: ChatAgentRequest):
    # simple echo bot for now
    msg = req.message.strip()
    reply = f"I received: {msg}. I can help with code edits, testing, or visualization."
    return {"reply": reply}


@app.post("/api/chat_agent_suggest_edit", response_model=SuggestEditResponse)
async def chat_agent_suggest_edit(req: SuggestEditRequest):
    code = CODE_STORE.get(req.sop_id)
    if not code:
        raise HTTPException(status_code=404, detail="sop_id not found or code not generated")
    main_file = code["main_file"]
    current = code["code_files"][0]["content"]

    suggestion = f"Apply instruction: {req.instruction}"
    # dummy patch model: replace first occurrence of 'log(' with added suffix
    patch = {
        "target_file": main_file,
        "find": "log(",
        "replace": 'log("[Edited] ", ',
        "count": 1,
    }
    return {"suggestion": suggestion, "patch": patch}


@app.post("/api/apply_code_edit", response_model=ApplyEditResponse)
async def apply_code_edit(req: ApplyEditRequest):
    code = CODE_STORE.get(req.sop_id)
    if not code:
        raise HTTPException(status_code=404, detail="sop_id not found or code not generated")

    patch = req.patch or {}
    target = patch.get("target_file") or code["main_file"]
    find = patch.get("find")
    replace = patch.get("replace")
    count = patch.get("count", 1)

    updated_files = []
    for f in code["code_files"]:
        if f["path"] == target and find:
            f["content"] = f["content"].replace(find, replace, count)
        updated_files.append({"path": f["path"], "content": f["content"]})

    CODE_STORE[req.sop_id]["code_files"] = updated_files
    return {"code_files": updated_files, "main_file": code["main_file"]}


@app.post("/api/explain_code_text", response_model=ExplainResponse)
async def explain_code_text(req: ExplainRequest):
    explanation = explain_code_with_llm(req.runtime_key, req.code, req.steps)
    logger.info(
        "explain_code_text: %s",
        "LLM" if client else "heuristic (Grok unavailable or key missing)"
    )
    return {"explanation": explanation}


# -----------------------------
# Shutdown hook (close httpx client)
# -----------------------------
@app.on_event("shutdown")
def _close_http_client():
    try:
        if _http_client:
            _http_client.close()
    except Exception:
        pass
