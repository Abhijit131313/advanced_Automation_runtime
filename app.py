# app.py
import os
import json
import uuid
import logging
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- Logging ---------------------------------------------------------------

app_logger = logging.getLogger("workbench-backend")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# ---- LLM (Grok via OpenAI-compatible SDK) ---------------------------------

# IMPORTANT: we use the OpenAI client pointed at xAI Grok API via env vars
from openai import OpenAI

GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5")

_llm_available = bool(GROK_API_KEY.strip())

client = OpenAI(
    api_key=GROK_API_KEY if _llm_available else None,
    base_url=GROK_API_URL
)

# ---- FastAPI ---------------------------------------------------------------

app = FastAPI(title="Workbench Studio Backend", version="0.1.0")

# Allow Bolt/localhost/Render frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Bolt origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- In-memory store -------------------------------------------------------

class SuggestedRuntime(BaseModel):
    key: str
    name: str
    confidence: float
    reason: str

class SopRecord(BaseModel):
    sop_id: str
    title: str
    text: str
    steps: List[str]
    suggested_runtimes: List[SuggestedRuntime]

SOP_STORE: Dict[str, SopRecord] = {}
CODE_STORE: Dict[str, Dict[str, Any]] = {}  # keyed by sop_id + runtime_key

# ---- Helpers ---------------------------------------------------------------

def safe_steps_from_text(text: str) -> List[str]:
    """
    Very simple step extractor: numbered/bulleted lines or sentences.
    """
    lines = [l.strip() for l in text.replace("\r", "").split("\n") if l.strip()]
    steps: List[str] = []
    for line in lines:
        if line[:2].isdigit() or line[:1] in {"-", "•", "*"} or line.lower().startswith("step "):
            # trim common prefixes
            cleaned = line
            cleaned = cleaned.lstrip("-•*").strip()
            cleaned = cleaned.split(":", 1)[-1].strip() if ":" in cleaned and cleaned.lower().startswith("step") else cleaned
            steps.append(cleaned)
        else:
            # fallback: treat line as a step if it reads like a sentence
            if len(line.split()) > 3:
                steps.append(line)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in steps:
        if s not in seen:
            unique.append(s); seen.add(s)
    return unique[:20]  # keep it bounded

def runtime_suggestions(steps: List[str]) -> List[SuggestedRuntime]:
    # Very rough heuristic ranking
    n = len(steps)
    has_human_review = any("review" in s.lower() or "officer" in s.lower() for s in steps)
    has_services = any(k in " ".join(steps).lower() for k in ["api", "queue", "s3", "kafka", "http"])
    has_waits = any(w in " ".join(steps).lower() for w in ["wait", "retry", "schedule", "timeout", "cron"])

    candidates = [
        SuggestedRuntime(key="bpmn", name="BPMN", confidence=0.5, reason="BPMN (XML) - good for human tasks & approvals"),
        SuggestedRuntime(key="camel", name="CAMEL", confidence=0.5, reason="Apache Camel (Java DSL) - good for API/adapter integration"),
        SuggestedRuntime(key="temporal", name="TEMPORAL", confidence=0.55, reason="Temporal (Java/TS) - good for durable, code-first workflows"),
    ]
    # adjust confidence
    for c in candidates:
        if c.key == "bpmn" and has_human_review:
            c.confidence += 0.15
        if c.key == "camel" and has_services:
            c.confidence += 0.15
        if c.key == "temporal" and has_waits:
            c.confidence += 0.15
        c.confidence = round(min(c.confidence, 0.95), 2)

    # sort desc
    candidates.sort(key=lambda x: x.confidence, reverse=True)
    return candidates[:3]

def summarize_sop(steps: List[str]) -> str:
    automated_guess = [s for s in steps if any(k in s.lower() for k in ["log", "update", "check", "archive", "send", "generate", "verify"])]
    manual_guess = [s for s in steps if any(k in s.lower() for k in ["review", "approve", "call", "inspect", "manual", "officer"])]
    auto_n = len(automated_guess)
    man_n = len(manual_guess)
    total = len(steps)
    return f"Detected {total} steps. Will automate {auto_n}; {max(total - auto_n, 0)} remain manual."

def _escape_braces(s: str) -> str:
    """
    Escape braces for format()/f-strings when embedding languages like Java/YAML.
    """
    return s.replace("{", "{{").replace("}", "}}")

# ---- Code Generators (safe braces) ----------------------------------------

def generate_bpmn_xml(title: str, steps: List[str]) -> str:
    tasks_xml = "\n".join([
        f'  <bpmn:task id="Task_{i+1}" name="{_escape_braces(steps[i])}"/>' for i in range(len(steps))
    ])
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" id="Definitions_workflow">
  <bpmn:process id="workflow" isExecutable="true">
{tasks_xml}
  </bpmn:process>
</bpmn:definitions>
"""
    return xml

def generate_camel_java_dsl(title: str, steps: List[str]) -> str:
    class_name = "RouteBuilder_" + uuid.uuid4().hex[:6]
    route_steps = ""
    for i, s in enumerate(steps):
        # simple log step to keep valid Camel DSL
        route_steps += f'            .to("log:step{i+1}") // {_escape_braces(s)}\n'

    content = (
        "package com.example.routes;\n\n"
        "import org.apache.camel.builder.RouteBuilder;\n"
        "import org.springframework.stereotype.Component;\n\n"
        "@Component\n"
        f"public class {class_name} extends RouteBuilder {{\n"
        "    @Override\n"
        "    public void configure() throws Exception {\n"
        f'        from("direct:workbench_routebuilder_{class_name.lower()}")\n'
        f"{route_steps}"
        "        ;\n"
        "    }\n"
        "}\n"
    )
    return content

def generate_temporal_java(title: str, steps: List[str]) -> str:
    # Build interface & impl with properly escaped comments
    iface = (
        "package com.example.temporal;\n\n"
        "import io.temporal.workflow.Workflow;\n\n"
        f"public interface IWorkflow{uuid.uuid4().hex[:6]} {{\n"
        "    void execute();\n"
        "}\n"
    )

    impl_class = f"WorkflowImpl{uuid.uuid4().hex[:6]}"
    activities_iface = "Activities"
    calls = ""
    acts = ""
    for i, s in enumerate(steps):
        calls += f"        activities.executeStep{i+1}();\n"
        acts += f"    // Activity {i+1}: {_escape_braces(s)}\n"

    impl = (
        "/* Implementation */\n"
        f"public class {impl_class} implements {iface.split('interface ')[1].split(' ')[0]} {{\n"
        f"    private final {activities_iface} activities = Workflow.newActivityStub({activities_iface}.class);\n\n"
        "    @Override\n"
        "    public void execute() {\n"
        f"{calls}"
        "    }\n"
        "}\n"
    )
    activities = (
        "/* Activities interface (skeleton) */\n"
        "interface Activities {\n"
        f"{acts}"
        "}\n"
    )
    return iface + "\n" + impl + "\n" + activities

def generate_knative_yaml(title: str, steps: List[str]) -> str:
    # Minimal Knative Service skeleton with annotations describing steps
    annotations = "\\n".join([f"- {_escape_braces(s)}" for s in steps])
    yaml = f"""apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: workbench-{uuid.uuid4().hex[:5]}
  annotations:
    workbench/steps: |
      {annotations}
spec:
  template:
    spec:
      containers:
        - image: gcr.io/demo/worker:latest
          env:
            - name: SCENARIO_TITLE
              value: "{_escape_braces(title)}"
"""
    return yaml

def pick_editor_mode(runtime_key: str) -> str:
    return {
        "bpmn": "xml",
        "camel": "java",
        "temporal": "java",
        "knative": "yaml",
    }.get(runtime_key, "text")

# ---- Schemas for API ------------------------------------------------------

class UploadSopResponse(BaseModel):
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
    editor_mode: str
    ui_schema: Dict[str, Any]
    confidence: float

# ---- Endpoints ------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/sop/upload", response_model=UploadSopResponse)
async def upload_sop(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    scenario_id: Optional[str] = Form(None),
):
    content = text or ""
    if file is not None:
        # Read as text safely
        try:
            raw = await file.read()
            try:
                content = raw.decode("utf-8")
            except Exception:
                content = "<binary or non-utf8 document>"
        except Exception:
            content = "<binary or non-utf8 document>"

    if not content.strip():
        content = "<binary or non-utf8 document>"

    sop_id = str(uuid.uuid4())
    steps = safe_steps_from_text(content)
    title = scenario_id or f"scenario-{sop_id}"

    suggestions = runtime_suggestions(steps)
    summary = summarize_sop(steps)

    SOP_STORE[sop_id] = SopRecord(
        sop_id=sop_id,
        title=title,
        text=content,
        steps=steps or ["User task"],
        suggested_runtimes=suggestions,
    )

    app_logger.info(f"Uploaded SOP {sop_id} title={title} steps={len(steps)}")

    automated_guess = [s for s in steps if any(k in s.lower() for k in ["log", "update", "check", "archive", "send", "generate", "verify"])]
    manual_guess = [s for s in steps if any(k in s.lower() for k in ["review", "approve", "call", "inspect", "manual", "officer"])]

    return UploadSopResponse(
        sop_id=sop_id,
        summary=summary,
        automated_actions=automated_guess or ["<binary or non-utf8 document>"],
        manual_actions=manual_guess,
        suggested_runtimes=suggestions
    )

@app.post("/api/generate_code", response_model=GenerateCodeResponse)
async def generate_code(req: GenerateCodeRequest):
    sop = SOP_STORE.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")

    title = sop.title
    steps = sop.steps

    runtime_key = req.runtime_key.lower().strip()
    if runtime_key not in {"bpmn", "camel", "temporal", "knative"}:
        # default to first suggestion
        runtime_key = (sop.suggested_runtimes[0].key if sop.suggested_runtimes else "bpmn")

    # Generate content
    if runtime_key == "bpmn":
        content = generate_bpmn_xml(title, steps)
        main_file = "main.xml"
    elif runtime_key == "camel":
        content = generate_camel_java_dsl(title, steps)
        main_file = "RouteBuilder.java"
    elif runtime_key == "temporal":
        content = generate_temporal_java(title, steps)
        main_file = "main.java"
    elif runtime_key == "knative":
        content = generate_knative_yaml(title, steps)
        main_file = "service.yaml"
    else:
        content = "\n".join(steps)
        main_file = "main.txt"

    editor_mode = pick_editor_mode(runtime_key)

    CODE_STORE[req.sop_id] = {
        "runtime_key": runtime_key,
        "main_file": main_file,
        "files": [
            {"path": main_file, "content": content}
        ],
        "confidence": next((r.confidence for r in sop.suggested_runtimes if r.key == runtime_key), 0.7)
    }

    app_logger.info(
        f"Generated code for sop_id={req.sop_id} runtime={runtime_key}"
        + (" (explain: heuristic)" if not _llm_available else "")
    )

    return GenerateCodeResponse(
        code_files=[CodeFile(**f) for f in CODE_STORE[req.sop_id]["files"]],
        main_file=main_file,
        editor_mode=editor_mode,
        ui_schema={"preview": True},
        confidence=CODE_STORE[req.sop_id]["confidence"]
    )

# ---- Visualization & Test -------------------------------------------------

class VisualizeRequest(BaseModel):
    sop_id: str
    runtime_key: Optional[str] = None
    code: Optional[str] = None

@app.post("/api/visualize_code")
async def visualize_code(req: VisualizeRequest):
    """
    Return a simple nodes/edges graph (heuristic).
    """
    sop = SOP_STORE.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")

    steps = sop.steps
    nodes = [{"id": f"n{i+1}", "label": s} for i, s in enumerate(steps)]
    edges = [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(1, len(steps))]

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {"count": len(nodes)}
    }

class RunTestRequest(BaseModel):
    sop_id: str
    inputs: Optional[Dict[str, Any]] = {}

@app.post("/api/run_test")
async def run_test(req: RunTestRequest):
    """
    A stub test runner that always returns success with a trace of steps.
    """
    sop = SOP_STORE.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")

    trace = [{"step": i+1, "action": s, "status": "ok"} for i, s in enumerate(sop.steps[:20])]
    return {
        "status": "success",
        "summary": f"Executed {len(trace)} steps.",
        "trace": trace
    }

# ---- Agent/Chat (stubs) ---------------------------------------------------

class ChatAgentRequest(BaseModel):
    sop_id: Optional[str] = None
    messages: List[Dict[str, str]]

@app.post("/api/chat_agent")
async def chat_agent(req: ChatAgentRequest):
    if not req.messages:
        return {"code": "400", "error": "Bad data: Messages cannot be empty"}
    # Simple echo / future LLM hook
    last = req.messages[-1].get("content", "")
    return {"reply": f"Noted. You said: {last}"}

class SuggestEditRequest(BaseModel):
    sop_id: str
    main_file: str
    code: str
    instruction: str

@app.post("/api/chat_agent_suggest_edit")
async def chat_agent_suggest_edit(req: SuggestEditRequest):
    # trivial suggestion stub
    suggestion = req.code + "\n// Suggested: Consider adding error handling."
    return {"suggested_code": suggestion}

class ApplyEditRequest(BaseModel):
    sop_id: str
    main_file: str
    new_code: str

@app.post("/api/apply_code_edit")
async def apply_code_edit(req: ApplyEditRequest):
    record = CODE_STORE.get(req.sop_id)
    if not record:
        raise HTTPException(status_code=404, detail="sop_id not found")
    for f in record["files"]:
        if f["path"] == req.main_file:
            f["content"] = req.new_code
            break
    else:
        record["files"].append({"path": req.main_file, "content": req.new_code})
    return {"status": "ok"}

# ---- Code Explanation (Grok) ----------------------------------------------

class ExplainRequest(BaseModel):
    code: str
    runtime_key: Optional[str] = None
    title: Optional[str] = None
    confidence: Optional[float] = None

def heuristic_code_explainer(code: str) -> str:
    # Fallback summary if Grok isn’t configured or fails
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    bullets = []
    for l in lines:
        if any(k in l.lower() for k in ["task", "log", "step", "activity", "process", "route", "service"]):
            bullets.append(l.strip("//").strip())
    if not bullets:
        bullets = lines[:10]
    joined = "\n\n".join(bullets[:10])
    return joined or "This workflow orchestrates a sequence of business steps."

@app.post("/api/explain_code_text")
async def explain_code_text(req: ExplainRequest):
    code = (req.code or "").strip()
    if not code:
        return {"explanation": "No code provided."}

    # Prefer Grok if available
    if _llm_available:
        try:
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert workflow/automation architect. "
                            "Given the code below (BPMN XML / Java / YAML / Camel DSL / Temporal), "
                            "explain concisely **what this automation does** as a bullet list for a process designer. "
                            "Avoid implementation details like imports; focus on the business flow."
                        ),
                    },
                    {"role": "user", "content": code},
                ],
                temperature=0.2,
            )
            explanation = resp.choices[0].message.content.strip()
            app_logger.info("explain_code_text: Grok model response")
            return {"explanation": explanation}
        except Exception as e:
            app_logger.warning(f"explain_code_text: heuristic (Grok error: {e})")
            return {"explanation": heuristic_code_explainer(code)}
    else:
        app_logger.info("explain_code_text: heuristic (Groq unavailable or key missing)")
        return {"explanation": heuristic_code_explainer(code)}

# ---- Root 404 guard -------------------------------------------------------

@app.get("/")
async def root():
    raise HTTPException(status_code=404, detail="Not Found")
