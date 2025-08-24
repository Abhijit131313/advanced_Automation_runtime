# app.py
import os
import io
import json
import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

import aiohttp
import asyncio
import yaml

# -----------------------------------------------------------------------------
# App setup & logger
# -----------------------------------------------------------------------------
app = FastAPI(title="Workbench Studio Backend", version="0.1.0")
logger = logging.getLogger("workbench-backend")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # scope-reduce for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# In-memory stores (simple demo persistence)
# -----------------------------------------------------------------------------
# sop_store[sop_id] = {
#   "title": str,
#   "text": str,             # parsed text
#   "steps": [str, ...],     # recognized steps
#   "suggested_runtimes": [...],
# }
sop_store: Dict[str, Dict[str, Any]] = {}

# generated_code_store[sop_id] = {
#   "runtime": str,
#   "code_files": [{"path": str, "content": str}],
#   "main_file": str,
#   "editor_mode": str,
#   "ui_schema": dict,
#   "confidence": float,
# }
generated_code_store: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Config (Grok / xAI)
# -----------------------------------------------------------------------------
GROK_API_KEY = os.getenv("GROK_API_KEY", "").strip()
GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1/chat/completions").strip()
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5").strip()

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class UploadSOPResponse(BaseModel):
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

class RunTestRequest(BaseModel):
    sop_id: str
    code: Optional[str] = None
    runtime: Optional[str] = None
    input: Optional[Dict[str, Any]] = None

class RunTestResponse(BaseModel):
    ok: bool
    logs: List[str]
    output: Dict[str, Any]

class VisualizeRequest(BaseModel):
    sop_id: str
    code: Optional[str] = None
    runtime: Optional[str] = None

class VisualizeResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class ChatAgentRequest(BaseModel):
    sop_id: Optional[str] = None
    messages: List[Dict[str, str]]

class ChatAgentResponse(BaseModel):
    reply: str

class SuggestEditRequest(BaseModel):
    sop_id: Optional[str] = None
    code: str
    instruction: str

class SuggestEditResponse(BaseModel):
    code: str

class ApplyCodeEditRequest(BaseModel):
    code: str
    patch: str  # future use

class ApplyCodeEditResponse(BaseModel):
    code: str

class ExplainRequest(BaseModel):
    sop_id: str
    runtime: Optional[str] = None

class ExplainResponse(BaseModel):
    sop_id: str
    runtime: str
    explanation: str

# -----------------------------------------------------------------------------
# Utility: SOP parsing & runtime suggestion
# -----------------------------------------------------------------------------
def _safe_text_from_upload(fobj: UploadFile) -> str:
    try:
        raw = fobj.file.read()
        try:
            return raw.decode("utf-8")
        except Exception:
            # Not valid UTF-8; return marker
            return "<binary or non-utf8 document>"
    finally:
        try:
            fobj.file.close()
        except Exception:
            pass

def _extract_steps(text: str) -> List[str]:
    """
    Naive step extractor: split on newlines and pick non-empty items.
    """
    if not text or text.strip() == "<binary or non-utf8 document>":
        return ["Receive input", "Process", "Finish"]
    lines = [ln.strip(" •-").strip() for ln in text.splitlines()]
    steps = [ln for ln in lines if ln and len(ln) > 2]
    # keep first ~10
    return steps[:10] or ["Receive input", "Process", "Finish"]

def _suggest_runtimes(steps: List[str]) -> List[Dict[str, Any]]:
    # Simple heuristic suggestions & confidences
    candidates = [
        {"key": "temporal", "name": "TEMPORAL", "confidence": 0.55, "reason": "Temporal (Java/TS) - good for code-first durable workflows"},
        {"key": "bpmn", "name": "BPMN", "confidence": 0.50, "reason": "BPMN (XML) - good for human tasks & manual approvals"},
        {"key": "camel", "name": "CAMEL", "confidence": 0.50, "reason": "Apache Camel (Java DSL) - good for API/adapter integration"},
    ]
    # Sort by confidence desc
    return sorted(candidates, key=lambda x: x["confidence"], reverse=True)

def _summarize_sop(steps: List[str]) -> Tuple[str, List[str], List[str]]:
    # For demo, designate all steps as "automated_actions"; none manual
    summary = f"Detected {len(steps)} steps. Will automate {len(steps)}; 0 remain manual."
    return summary, steps, []

# -----------------------------------------------------------------------------
# Code generators (very safe strings; avoid str.format brace issues)
# -----------------------------------------------------------------------------
def generate_bpmn_xml(title: str, steps: List[str]) -> str:
    """
    Very minimal BPMN XML with tasks only (no flows for brevity).
    """
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" id="Definitions_workflow">',
        f'  <bpmn:process id="workflow" isExecutable="true">',
    ]
    for i, st in enumerate(steps, 1):
        safe = st.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f'    <bpmn:task id="Task_{i}" name="{safe}" />')
    lines += [
        "  </bpmn:process>",
        "</bpmn:definitions>",
    ]
    return "\n".join(lines)

def generate_camel_java_dsl(title: str, steps: List[str]) -> str:
    route_steps = []
    for i, st in enumerate(steps, 1):
        msg = st.replace('"', '\\"')
        route_steps.append(f'            .to("log:step{i}?showAll=true&multiline=true") // {msg}')
    route = "\n".join(route_steps) if route_steps else '            .to("log:empty")'

    return (
        "package com.example.routes;\n\n"
        "import org.apache.camel.builder.RouteBuilder;\n"
        "import org.springframework.stereotype.Component;\n\n"
        "@Component\n"
        "public class RouteBuilder_" + uuid.uuid4().hex[:6] + " extends RouteBuilder {\n"
        "    @Override\n"
        "    public void configure() throws Exception {\n"
        '        from("direct:workbench_route")\n'
        f"{route}\n"
        "        ;\n"
        "    }\n"
        "}\n"
    )

def generate_temporal_java(title: str, steps: List[str]) -> str:
    # Interface & implementation with Activities stub
    iface = (
        "package com.example.temporal;\n\n"
        "import io.temporal.workflow.Workflow;\n\n"
        "public interface IWorkflow" + uuid.uuid4().hex[:5] + " {\n"
        "    void execute();\n"
        "}\n"
    )
    acts_methods = []
    for i, st in enumerate(steps, 1):
        acts_methods.append(f"    void executeStep{i}(); // {st}")
    activities = (
        "interface Activities {\n" +
        "\n".join(acts_methods) + "\n}\n"
    )
    impl_calls = []
    for i, _ in enumerate(steps, 1):
        impl_calls.append(f"        activities.executeStep{i}();")
    impl = (
        "/* Implementation */\n"
        "public class WorkflowImpl" + uuid.uuid4().hex[:6] + " implements IWorkflow" + uuid.uuid4().hex[:5] + " {\n"
        "    private final Activities activities = Workflow.newActivityStub(Activities.class);\n"
        "    @Override\n"
        "    public void execute() {\n" +
        "\n".join(impl_calls) + "\n"
        "    }\n"
        "}\n"
    )
    return "\n".join([iface, "/* Activities interface (skeleton) */", activities, impl])

def generate_knative_yaml(title: str, steps: List[str]) -> str:
    # toy Knative Service
    doc = {
        "apiVersion": "serving.knative.dev/v1",
        "kind": "Service",
        "metadata": {"name": f"workbench-{uuid.uuid4().hex[:6]}"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "image": "ghcr.io/example/workbench:latest",
                        "env": [{"name": f"STEP_{i}", "value": s} for i, s in enumerate(steps, 1)]
                    }]
                }
            }
        }
    }
    return yaml.safe_dump(doc, sort_keys=False)

def pick_editor_mode(runtime_key: str) -> str:
    mapping = {
        "bpmn": "xml",
        "camel": "java",
        "temporal": "java",
        "knative": "yaml",
    }
    return mapping.get(runtime_key.lower(), "text")

# -----------------------------------------------------------------------------
# Grok (xAI) explainer
# -----------------------------------------------------------------------------
async def grok_explain(runtime: str, code: str) -> str:
    """
    Call xAI Chat Completions API. Falls back to heuristic on errors.
    """
    sys_prompt = (
        "You are an automation analyst. Given the code below and the chosen runtime, "
        "write a concise, plain-English explanation (bulleted where helpful) that a process "
        "designer can understand. Avoid jargon. Keep it under ~180 words."
    )
    user_prompt = f"Runtime: {runtime}\n\nCode:\n```\n{code}\n```"

    if not GROK_API_KEY:
        logger.info("explain_code_text: heuristic (Groq unavailable or key missing)")
        return heuristic_explanation(runtime, code)

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": DEFAULT_MODEL or "gpt-5",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            }
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json",
            }
            async with session.post(GROK_API_URL, headers=headers, json=payload, timeout=60) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    logger.warning(f"grok explain non-200: {resp.status} {txt}")
                    return heuristic_explanation(runtime, code)
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
    except Exception as e:
        logger.info(f"explain_code_text: grok failed ({e}); falling back to heuristic")
        return heuristic_explanation(runtime, code)

def heuristic_explanation(runtime: str, code: str) -> str:
    bullets = []
    if runtime.lower() == "bpmn":
        bullets = [
            "Parses a BPMN process with user tasks for each step.",
            "Each task represents a discrete activity in the business flow.",
            "Intended for orchestration and human-in-the-loop approvals where needed.",
            "This XML can be viewed and refined in a BPMN modeler."
        ]
    elif runtime.lower() == "camel":
        bullets = [
            "Defines an Apache Camel RouteBuilder pipeline.",
            "Consumes from a direct endpoint and logs each step.",
            "Good for system-to-system integration and adapters.",
            "You can replace log steps with real components (HTTP, JMS, DB, etc.)."
        ]
    elif runtime.lower() == "temporal":
        bullets = [
            "Defines a Temporal workflow interface and implementation.",
            "Each step is an Activity invocation (reliable, retryable).",
            "Temporal persists state and handles retries/durable execution.",
            "Implement Activities to call real services or perform tasks."
        ]
    else:
        bullets = [
            "Represents an automation workflow for the selected runtime.",
            "Each step maps to a discrete unit of work.",
            "You can customize handlers to integrate with real systems."
        ]
    return "This automation defines a workflow in the selected runtime.\n\n" + "\n".join([f"- {b}" for b in bullets])

# -----------------------------------------------------------------------------
# Core: Generate code from SOP & runtime
# -----------------------------------------------------------------------------
def generate_code_from_sop(sop_data: Dict[str, Any], runtime_key: str) -> Tuple[Dict[str, Any], str]:
    title = sop_data.get("title") or "workflow"
    steps = sop_data.get("steps") or ["Receive input", "Process", "Finish"]

    runtime = runtime_key.lower()
    if runtime == "bpmn":
        content = generate_bpmn_xml(title, steps)
        main_file = "main.xml"
    elif runtime == "camel":
        content = generate_camel_java_dsl(title, steps)
        main_file = "RouteBuilder.java"
    elif runtime == "temporal":
        content = generate_temporal_java(title, steps)
        main_file = "main.java"
    elif runtime == "knative":
        content = generate_knative_yaml(title, steps)
        main_file = "service.yaml"
    else:
        content = "\n".join(steps)
        main_file = "steps.txt"

    editor_mode = pick_editor_mode(runtime)
    code_files = [{"path": main_file, "content": content}]

    # trivial UI schema + confidence
    ui_schema = {
        "runtime": runtime.upper(),
        "nodes": [{"id": f"task_{i}", "label": s} for i, s in enumerate(steps, 1)],
        "edges": [{"from": f"task_{i}", "to": f"task_{i+1}"} for i in range(1, len(steps))]
    }
    confidence = 0.65 if runtime in ("bpmn", "temporal") else 0.55

    payload = {
        "code_files": code_files,
        "main_file": main_file,
        "editor_mode": editor_mode,
        "ui_schema": ui_schema,
        "confidence": confidence
    }
    return payload, editor_mode

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"ok": True}

@app.post("/api/sop/upload", response_model=UploadSOPResponse)
async def upload_sop(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    scenario_id: Optional[str] = Form(None),
):
    # Accept either explicit text or uploaded file
    sop_text = ""
    if text and text.strip():
        sop_text = text
    elif file is not None:
        sop_text = _safe_text_from_upload(file)
    else:
        sop_text = "No SOP text provided."

    steps = _extract_steps(sop_text)
    suggested = _suggest_runtimes(steps)
    summary, automated, manual = _summarize_sop(steps)

    sop_id = str(uuid.uuid4())
    sop_store[sop_id] = {
        "title": scenario_id or f"scenario-{sop_id}",
        "text": sop_text,
        "steps": steps,
        "suggested_runtimes": suggested,
    }

    logger.info(f"Uploaded SOP {sop_id} title={sop_store[sop_id]['title']} steps={len(steps)}")
    return UploadSOPResponse(
        sop_id=sop_id,
        summary=summary,
        automated_actions=automated,
        manual_actions=manual,
        suggested_runtimes=suggested,
    )

@app.post("/api/generate_code", response_model=GenerateCodeResponse)
async def generate_code(req: GenerateCodeRequest):
    sop_id = req.sop_id
    runtime_key = (req.runtime_key or "bpmn").lower()

    if sop_id not in sop_store:
        raise HTTPException(status_code=404, detail="sop_id not found")

    sop_data = sop_store[sop_id]
    payload, editor_mode = generate_code_from_sop(sop_data, runtime_key)

    generated_code_store[sop_id] = {
        "runtime": runtime_key,
        **payload
    }

    logger.info(f"Generated code for sop_id={sop_id} runtime={runtime_key} (explain: heuristic)")
    return GenerateCodeResponse(**payload)

@app.post("/api/run_test", response_model=RunTestResponse)
async def run_test(req: RunTestRequest):
    # Simulated test runner
    sop_id = req.sop_id
    runtime = (req.runtime or "bpmn").lower()
    if not req.code:
        # use stored if available
        if sop_id in generated_code_store:
            # find main file content
            cf = generated_code_store[sop_id]["code_files"]
            code = next((c["content"] for c in cf if c["path"] == generated_code_store[sop_id]["main_file"]), cf[0]["content"])
        else:
            code = ""
    else:
        code = req.code

    logs = [
        f"Runtime: {runtime}",
        "Compiling (simulated)... OK",
        "Running workflow (simulated)... OK",
        "All steps executed."
    ]
    return RunTestResponse(ok=True, logs=logs, output={"result": "success"})

@app.post("/api/visualize_code", response_model=VisualizeResponse)
async def visualize_code(req: VisualizeRequest):
    sop_id = req.sop_id
    if sop_id not in sop_store:
        raise HTTPException(status_code=404, detail="sop_id not found")

    steps = sop_store[sop_id].get("steps", [])
    nodes = [{"id": f"n{i}", "label": s} for i, s in enumerate(steps, 1)]
    edges = [{"from": f"n{i}", "to": f"n{i+1}"} for i in range(1, len(steps))]
    return VisualizeResponse(nodes=nodes, edges=edges)

@app.post("/api/chat_agent", response_model=ChatAgentResponse)
async def chat_agent(req: ChatAgentRequest):
    # Lightweight echo assistant for now
    if not req.messages:
        return ChatAgentResponse(reply="Please provide a question.")
    last = req.messages[-1].get("content", "")
    return ChatAgentResponse(reply=f"I read: {last}\nTry adjusting step 2 to call a real service.")

@app.post("/api/chat_agent_suggest_edit", response_model=SuggestEditResponse)
async def chat_agent_suggest_edit(req: SuggestEditRequest):
    # Naive suggestion example: add a comment to top
    suggestion = "// Suggested: add validation before step 1\n" + req.code
    return SuggestEditResponse(code=suggestion)

@app.post("/api/apply_code_edit", response_model=ApplyCodeEditResponse)
async def apply_code_edit(req: ApplyCodeEditRequest):
    # No real patching here
    return ApplyCodeEditResponse(code=req.code)

# -------------------- NEW / UPDATED --------------------

@app.post("/api/explain_code_text", response_model=ExplainResponse)
async def explain_code_text(req: ExplainRequest):
    """
    Always return an explanation:
    - If generated code missing for sop_id, auto-generate using requested or default runtime.
    - Then explain via Grok; if Grok unavailable, return heuristic.
    """
    sop_id = req.sop_id
    runtime = (req.runtime or "bpmn").lower()

    if sop_id not in sop_store:
        raise HTTPException(status_code=404, detail="SOP not found. Please upload it first.")

    # If we don't have code yet, auto-generate it
    if sop_id not in generated_code_store:
        sop_data = sop_store[sop_id]
        payload, _ = generate_code_from_sop(sop_data, runtime)
        generated_code_store[sop_id] = {
            "runtime": runtime,
            **payload
        }
        logger.info(f"Auto-generated code for explanation sop_id={sop_id} runtime={runtime}")

    code_entry = generated_code_store[sop_id]
    # Find main file content
    main_path = code_entry["main_file"]
    main_content = next(
        (c["content"] for c in code_entry["code_files"] if c["path"] == main_path),
        code_entry["code_files"][0]["content"]
    )

    explanation = await grok_explain(runtime, main_content)
    logger.info(f"explain_code_text: delivered explanation via {'grok' if GROK_API_KEY else 'heuristic'}")
    return ExplainResponse(sop_id=sop_id, runtime=runtime, explanation=explanation)

# -----------------------------------------------------------------------------
# Root (optional)
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    return JSONResponse({"detail": "Workbench Studio Backend is running"})
