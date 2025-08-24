# app.py
import os
import io
import json
import uuid
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional Groq client (installed via requirements.txt)
# We call the REST endpoint via httpx so we don't depend on SDK behaviors.
import httpx

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("workbench-backend")

# --------------------------------------------------------------------------------------
# App & CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="Workbench Studio Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bolt dev and prod will hit from various origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# In-memory stores (simple PoC state)
# --------------------------------------------------------------------------------------
SOP_STORE: Dict[str, Dict[str, Any]] = {}
CODE_STORE: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------------------------------------------
# Environment & Grok config
# --------------------------------------------------------------------------------------
GROK_API_KEY = os.getenv("GROK_API_KEY", "").strip()
GROK_API_URL = os.getenv("GROK_API_URL", "").strip() or "https://api.x.ai/v1"
GROK_MODEL = os.getenv("GROK_MODEL", "").strip() or "grok-2-latest"  # adjust if needed
# If you’re routing Groq requests via a proxy, set GROK_HTTP_PROXY (optional)
GROK_HTTP_PROXY = os.getenv("GROK_HTTP_PROXY", "").strip() or None

# --------------------------------------------------------------------------------------
# Pydantic models (v1-compatible)
# --------------------------------------------------------------------------------------
class SuggestedRuntime(BaseModel):
    key: str
    name: str
    confidence: float
    reason: str

class UploadSopResponse(BaseModel):
    sop_id: str
    summary: str
    automated_actions: List[str]
    manual_actions: List[str]
    suggested_runtimes: List[SuggestedRuntime]

class CodeFile(BaseModel):
    path: str
    content: str

class GenerateCodeRequest(BaseModel):
    sop_id: str
    runtime_key: str
    options: Optional[Dict[str, Any]] = None

class GenerateCodeResponse(BaseModel):
    code_files: List[CodeFile]
    main_file: str
    editor_mode: str
    ui_schema: Dict[str, Any]
    confidence: float

class RunTestRequest(BaseModel):
    sop_id: str
    runtime_key: str
    code_files: Optional[List[CodeFile]] = None
    inputs: Optional[Dict[str, Any]] = None

class RunTestResponse(BaseModel):
    ok: bool
    logs: List[str]
    result_preview: Optional[str] = None

class VisualizeCodeRequest(BaseModel):
    sop_id: Optional[str] = None
    runtime_key: Optional[str] = None
    code_files: Optional[List[CodeFile]] = None

class VisualizeCodeResponse(BaseModel):
    ok: bool
    kind: str
    payload: Dict[str, Any]

class ChatAgentRequest(BaseModel):
    sop_id: Optional[str] = None
    message: str

class ChatAgentResponse(BaseModel):
    reply: str

class ChatSuggestEditRequest(BaseModel):
    sop_id: str
    runtime_key: str
    code_files: List[CodeFile]
    message: str

class ChatSuggestEditResponse(BaseModel):
    suggested_files: List[CodeFile]
    note: Optional[str] = None

class ApplyCodeEditRequest(BaseModel):
    sop_id: str
    code_files: List[CodeFile]

class ApplyCodeEditResponse(BaseModel):
    ok: bool

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def _extract_text_from_upload(file: Optional[UploadFile], text_fallback: Optional[str]) -> str:
    """
    Minimal text extraction:
    - If text_fallback is provided, use it.
    - If PDF is uploaded, read bytes and do a naive decode to avoid extra deps.
      (Real apps should use pdfminer or pymupdf — omitted for dependency simplicity.)
    - Otherwise, return empty string.
    """
    if text_fallback and text_fallback.strip():
        return text_fallback.strip()

    if file and file.filename and file.content_type in ("application/pdf", "application/octet-stream"):
        try:
            # Just a naive attempt — some PDFs might include readable text.
            raw = file.file.read()
            try:
                # Try direct UTF-8 decode (may fail for binary PDFs)
                text = raw.decode("utf-8", errors="ignore")
            finally:
                file.file.close()
            # If nothing sensible, make a placeholder
            if not text or len(text.strip()) < 10:
                text = "Step 1: Ingest input\nStep 2: Validate record\nStep 3: Call external service\nStep 4: Update system\nStep 5: Notify user"
            return text
        except Exception:
            return "Step 1: Ingest input\nStep 2: Validate record\nStep 3: Call external service\nStep 4: Update system\nStep 5: Notify user"

    return (text_fallback or "").strip()

def _split_steps(s: str) -> List[str]:
    # Split by lines, filter empties; also accept "Step N:" style.
    lines = [ln.strip(" \t\r\n-•") for ln in s.splitlines()]
    steps = [ln for ln in lines if ln]
    # If user pasted a paragraph, try to split sentences.
    if len(steps) <= 1 and "." in s:
        steps = [seg.strip() for seg in s.split(".") if seg.strip()]
    # Hard-cap to something sane
    return steps[:50] if steps else ["Receive input", "Validate", "Process", "Persist", "Notify"]

def _suggest_runtimes_from_steps(steps: List[str]) -> List[SuggestedRuntime]:
    # Heuristic ranker
    # Simple scoring by keywords
    text = " ".join(steps).lower()
    scores = []
    def add(key, name, reason, base):
        scores.append((key, name, reason, base))
    # Heuristics
    base_temporal = 0.5 + (0.05 if "retry" in text or "wait" in text or "timeout" in text else 0)
    base_bpmn = 0.5 + (0.05 if "approve" in text or "review" in text or "human" in text else 0)
    base_camel = 0.5 + (0.05 if "api" in text or "route" in text or "transform" in text else 0)
    base_knative = 0.45 + (0.05 if "event" in text or "webhook" in text else 0)

    add("temporal", "TEMPORAL", "Temporal (code-first durable workflows)", base_temporal)
    add("bpmn", "BPMN", "BPMN (XML) — human tasks & approvals", base_bpmn)
    add("camel", "CAMEL", "Apache Camel (Java DSL) — routing/integration", base_camel)
    add("knative", "KNATIVE", "Knative (YAML) — event-driven serverless", base_knative)

    # Pick top 3 by score
    top = sorted(scores, key=lambda x: x[3], reverse=True)[:3]
    return [SuggestedRuntime(key=k, name=n, confidence=round(c, 2), reason=r) for (k, n, r, c) in top]

# --------------------------------------------------------------------------------------
# Code generators (templates with doubled braces for safe .format())
# --------------------------------------------------------------------------------------
def generate_bpmn_xml(title: str, steps: List[str]) -> str:
    task_xml = []
    for i, step in enumerate(steps, start=1):
        tid = f"task{i}"
        task_xml.append(
            f'    <bpmn:task id="{tid}" name="{step}"/>\n'
        )
    seq = []
    for i in range(1, len(steps)):
        seq.append(
            f'    <bpmn:sequenceFlow id="flow{i}" sourceRef="task{i}" targetRef="task{i+1}"/>\n'
        )
    # Double braces in template where literal braces are required
    template = """<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
                  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
                  xmlns:dc="http://www.omg.org/spec/DD/20100524/DC"
                  targetNamespace="http://example.com/bpmn">
  <bpmn:process id="Process_1" name="{title}">
{tasks}
{flows}
  </bpmn:process>
</bpmn:definitions>
""".format(title=title, tasks="".join(task_xml), flows="".join(seq))
    return template

def generate_camel_java_dsl(title: str, steps: List[str]) -> str:
    class_name = "".join([c for c in title.title() if c.isalnum()]) or "Automation"
    route_steps = []
    # Start from a direct route; chain steps as log() for PoC
    for i, step in enumerate(steps, start=1):
        if i == 1:
            route_steps.append('            from("direct:start")\n')
        prefix = "            " if i == 1 else "                "
        route_steps.append(f'{prefix}.log("Step {i}: {step}")\n')
    route_steps.append('                .to("mock:result");\n')

    # Escape braces for .format by doubling them
    template = """package com.example.camel;

import org.apache.camel.builder.RouteBuilder;

public class {cls}Route extends RouteBuilder {{
    @Override
    public void configure() throws Exception {{
{steps}    }}
}}
""".format(cls=class_name, steps="".join(route_steps))
    return template

def generate_temporal_java(title: str, steps: List[str]) -> str:
    # Create safe identifiers
    workflow_interface = f"IWorkflow{uuid.uuid4().hex[:6]}"
    workflow_impl = f"WorkflowImpl{uuid.uuid4().hex[:6]}"

    activity_iface = "Activities"
    # Build activity calls and interface comments
    activity_calls = []
    activity_comments = []
    for i, step in enumerate(steps, start=1):
        activity_calls.append(f"        activities.executeStep{i}();\n")
        activity_comments.append(f"    // Activity {i}: {step}\n")

    template = """package com.example.temporal;

import io.temporal.workflow.Workflow;

public interface {iface} {{
    void execute();
}}

/* Implementation */
public class {impl} implements {iface} {{
    private final Activities activities = Workflow.newActivityStub(Activities.class);

    @Override
    public void execute() {{
{calls}    }}
}}

/* Activities interface (skeleton) */
interface Activities {{
{acts}}}
""".format(iface=workflow_interface, impl=workflow_impl, calls="".join(activity_calls), acts="".join(activity_comments))
    return template

def generate_knative_yaml(title: str, steps: List[str]) -> str:
    # Minimal service and a comment block listing steps
    steps_block = "\n".join([f"# - {s}" for s in steps])
    template = """apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {name}
spec:
  template:
    spec:
      containers:
        - image: gcr.io/my-project/{name}:latest
          env:
            - name: WORKFLOW_TITLE
              value: "{title}"
# Steps (for reference):
{steps_block}
""".format(name=(title.lower().replace(" ", "-") or "automation"), title=title, steps_block=steps_block)
    return template

# --------------------------------------------------------------------------------------
# GROK explanation (LLM) with heuristic fallback
# --------------------------------------------------------------------------------------
def grok_explain_code_text(runtime_key: str, code_files: List[Dict[str, str]], sop_steps: Optional[List[str]]) -> str:
    """
    Calls GROK if configured; otherwise returns heuristic explanation.
    """
    # Build a compact prompt
    joined_files = []
    for f in code_files:
        path = f.get("path", "unknown")
        content = f.get("content", "")[:6000]  # keep short
        joined_files.append(f"FILE: {path}\n-----\n{content}\n")
    files_blob = "\n\n".join(joined_files)

    steps_blob = "\n".join([f"- {s}" for s in (sop_steps or [])]) if sop_steps else "(no SOP steps provided)"

    system_msg = (
        "You are a senior automation engineer. Explain clearly and concisely what the provided workflow code does, "
        "in bullet points suitable for a process designer. Mention the runtime (BPMN/Camel/Temporal/Knative) and "
        "summarize the execution flow and responsibilities."
    )

    user_msg = f"""Runtime: {runtime_key}
SOP Steps:
{steps_blob}

Code context:
{files_blob}

Return a short explanatory summary (5–12 bullets) and a one-paragraph overview.
"""

    # Try GROK first if a key is present
    if GROK_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": GROK_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.2,
            }
            proxies = {"http://": GROK_HTTP_PROXY, "https://": GROK_HTTP_PROXY} if GROK_HTTP_PROXY else None
            with httpx.Client(proxies=proxies, timeout=60.0) as client:
                resp = client.post(f"{GROK_API_URL.rstrip('/')}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                # x.ai style: choices[0].message.content
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if content:
                    log.info("explain_code_text: grok success")
                    return content
        except Exception as e:
            log.info(f"explain_code_text: grok failed ({e}); falling back to heuristic")

    # Heuristic fallback
    log.info("explain_code_text: heuristic (Groq unavailable or key missing)")
    bullets = []
    rk = (runtime_key or "").lower()
    if rk == "bpmn":
        bullets.append("Defines a BPMN process where each SOP step is modeled as a task.")
        bullets.append("The tasks are connected in sequence to represent the end-to-end flow.")
        bullets.append("Suitable for human approvals and manual checkpoints if added later.")
    elif rk == "camel":
        bullets.append("Defines an Apache Camel route starting from a direct endpoint.")
        bullets.append("Each SOP step is represented as a log/action in the route.")
        bullets.append("Good fit for integration, routing, and light transformations.")
    elif rk == "temporal":
        bullets.append("Defines a Temporal workflow interface and implementation.")
        bullets.append("Each SOP step is executed via an Activity call with durable orchestration.")
        bullets.append("Temporal handles retries/timeouts and ensures workflow durability.")
    elif rk == "knative":
        bullets.append("Defines a Knative Service resource (YAML) to run the workflow as serverless.")
        bullets.append("SOP steps are referenced for context and can be implemented in the container.")
        bullets.append("Suitable for event-driven triggers and autoscaling workloads.")
    else:
        bullets.append("Implements a linear workflow that executes steps in order.")

    if sop_steps:
        bullets.append(f"Includes {len(sop_steps)} steps derived from the SOP input.")
    else:
        bullets.append("SOP steps were not provided; explanation is based on code structure only.")

    para = "In summary, this automation implements a linear, step-by-step execution based on the SOP. The selected runtime shapes how tasks are orchestrated: BPMN emphasizes human/task modeling, Camel focuses on integration routes, Temporal provides durable code-first workflows with retries/state, and Knative packages the flow for event-driven serverless execution."
    return "\n".join([f"- {b}" for b in bullets]) + "\n\n" + para

# --------------------------------------------------------------------------------------
# API: Health
# --------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True}

# --------------------------------------------------------------------------------------
# API: Upload SOP (PDF or text) -> summary + suggested runtimes
# --------------------------------------------------------------------------------------
@app.post("/api/sop/upload", response_model=UploadSopResponse)
async def upload_sop(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    scenario_id: Optional[str] = Form(None),
):
    content = _extract_text_from_upload(file, text)
    steps = _split_steps(content)
    auto_count = len(steps)  # PoC assumption: all can be automated; UI still shows summary counts
    manual_count = max(0, len(steps) - auto_count)

    sop_id = str(uuid.uuid4())
    title = scenario_id or f"scenario-{uuid.uuid4().hex[:12]}"

    SOP_STORE[sop_id] = {
        "title": title,
        "raw_text": content,
        "steps": steps,
    }

    suggested = _suggest_runtimes_from_steps(steps)
    log.info(f"Uploaded SOP {sop_id} title={title} steps={len(steps)}")

    return UploadSopResponse(
        sop_id=sop_id,
        summary=f"Detected {len(steps)} steps. Will automate {auto_count}; {manual_count} remain manual.",
        automated_actions=["<binary or non-utf8 document>"] if (file and file.content_type == "application/pdf") else steps,
        manual_actions=[],
        suggested_runtimes=suggested,
    )

# --------------------------------------------------------------------------------------
# API: Generate Code for selected runtime
# --------------------------------------------------------------------------------------
@app.post("/api/generate_code", response_model=GenerateCodeResponse)
async def generate_code(req: GenerateCodeRequest):
    sop_id = req.sop_id
    runtime_key = req.runtime_key.lower().strip()

    if sop_id not in SOP_STORE:
        raise HTTPException(status_code=404, detail="sop_id not found")

    title = SOP_STORE[sop_id]["title"]
    steps: List[str] = SOP_STORE[sop_id]["steps"]

    if runtime_key == "bpmn":
        content = generate_bpmn_xml(title, steps)
        code_files = [CodeFile(path="workflow.bpmn.xml", content=content)]
        editor_mode = "xml"
        main_file = "workflow.bpmn.xml"
        conf = 0.65
    elif runtime_key == "camel":
        content = generate_camel_java_dsl(title, steps)
        code_files = [CodeFile(path=f"{title.title().replace(' ', '')}Route.java", content=content)]
        editor_mode = "java"
        main_file = code_files[0].path
        conf = 0.62
    elif runtime_key == "temporal":
        content = generate_temporal_java(title, steps)
        code_files = [CodeFile(path=f"{title.title().replace(' ', '')}Workflow.java", content=content)]
        editor_mode = "java"
        main_file = code_files[0].path
        conf = 0.68
    elif runtime_key == "knative":
        content = generate_knative_yaml(title, steps)
        code_files = [CodeFile(path="service.yaml", content=content)]
        editor_mode = "yaml"
        main_file = "service.yaml"
        conf = 0.58
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported runtime_key: {runtime_key}")

    # Store code for later visualize/test/explain
    CODE_STORE[sop_id] = {
        "runtime_key": runtime_key,
        "code_files": [cf.dict() for cf in code_files],
        "main_file": main_file,
        "editor_mode": editor_mode,
        "confidence": conf,
    }

    # Try to precompute explanation into CODE_STORE (non-blocking feel)
    try:
        explanation = grok_explain_code_text(runtime_key, CODE_STORE[sop_id]["code_files"], SOP_STORE[sop_id].get("steps"))
        CODE_STORE[sop_id]["explanation_text"] = explanation
        explain_note = "grok" if GROK_API_KEY else "heuristic"
        log.info(f"Generated code for sop_id={sop_id} runtime={runtime_key} (explain: {explain_note})")
    except Exception as e:
        log.info(f"explain precompute failed: {e}")

    return GenerateCodeResponse(
        code_files=code_files,
        main_file=main_file,
        editor_mode=editor_mode,
        ui_schema={},   # keep simple; Bolt will layout
        confidence=conf,
    )

# --------------------------------------------------------------------------------------
# API: Run Test (stub)
# --------------------------------------------------------------------------------------
@app.post("/api/run_test", response_model=RunTestResponse)
async def run_test(req: RunTestRequest):
    # Pull latest code from store if not supplied
    if not req.code_files and req.sop_id in CODE_STORE:
        code_files = CODE_STORE[req.sop_id].get("code_files", [])
    else:
        code_files = [cf.dict() for cf in (req.code_files or [])]

    logs = ["Starting test..."]
    for f in code_files:
        logs.append(f"Loaded file: {f.get('path')}")
    logs.append("Executing steps (simulated)... OK")
    return RunTestResponse(ok=True, logs=logs, result_preview="Test completed successfully (simulated).")

# --------------------------------------------------------------------------------------
# API: Visualize Code (simple visualization payload per runtime)
# --------------------------------------------------------------------------------------
@app.post("/api/visualize_code", response_model=VisualizeCodeResponse)
async def visualize_code(req: VisualizeCodeRequest):
    runtime_key = (req.runtime_key or "").lower()
    code_files = req.code_files

    if (not code_files or not isinstance(code_files, list)) and req.sop_id and req.sop_id in CODE_STORE:
        code_files = CODE_STORE[req.sop_id]["code_files"]

    if not runtime_key and req.sop_id and req.sop_id in CODE_STORE:
        runtime_key = CODE_STORE[req.sop_id]["runtime_key"]

    if not runtime_key or not code_files:
        return VisualizeCodeResponse(ok=False, kind="error", payload={"message": "Missing runtime_key or code_files"})

    # Extremely lightweight diagram model that the front end can render
    steps = []
    # Try to infer steps from SOP if available:
    if req.sop_id and req.sop_id in SOP_STORE:
        steps = SOP_STORE[req.sop_id]["steps"]

    nodes = [{"id": f"step{i}", "label": s} for i, s in enumerate(steps, start=1)]
    edges = [{"from": f"step{i}", "to": f"step{i+1}"} for i in range(1, len(steps))]

    payload = {
        "title": "Workflow",
        "runtime": runtime_key,
        "nodes": nodes,
        "edges": edges,
    }
    return VisualizeCodeResponse(ok=True, kind="graph", payload=payload)

# --------------------------------------------------------------------------------------
# API: Chat Agent (stub) and Suggest Edit (stub)
# --------------------------------------------------------------------------------------
@app.post("/api/chat_agent", response_model=ChatAgentResponse)
async def chat_agent(req: ChatAgentRequest):
    msg = (req.message or "").strip()
    if not msg:
        return ChatAgentResponse(reply="Please provide a message.")
    return ChatAgentResponse(reply=f"I received: '{msg}'. For now, I'm a stub; code editing is available via Suggest Edit.")

@app.post("/api/chat_agent_suggest_edit", response_model=ChatSuggestEditResponse)
async def chat_agent_suggest_edit(req: ChatSuggestEditRequest):
    # Naive example: append a comment to the first file
    if not req.code_files:
        raise HTTPException(status_code=400, detail="No code files provided")
    new_files = []
    for i, cf in enumerate(req.code_files):
        text = cf.content
        if i == 0:
            text += "\n// Suggested by agent: consider adding error handling here."
        new_files.append(CodeFile(path=cf.path, content=text))
    return ChatSuggestEditResponse(suggested_files=new_files, note="Added a small comment to the first file.")

@app.post("/api/apply_code_edit", response_model=ApplyCodeEditResponse)
async def apply_code_edit(req: ApplyCodeEditRequest):
    if req.sop_id not in CODE_STORE:
        raise HTTPException(status_code=404, detail="sop_id not found")
    CODE_STORE[req.sop_id]["code_files"] = [cf.dict() for cf in req.code_files]
    return ApplyCodeEditResponse(ok=True)

# --------------------------------------------------------------------------------------
# API: Explain Code (TEXT) — LENIENT version (fixes 422 from Bolt)
# --------------------------------------------------------------------------------------
@app.post("/api/explain_code_text")
async def explain_code_text(req: Request):
    """
    Lenient handler: accepts either snake_case or camelCase, and can infer
    missing values from in-memory SOP_STORE/CODE_STORE to avoid 422s.
    """
    try:
        data = await req.json()
    except Exception:
        data = {}

    # Accept snake_case or camelCase
    sop_id = data.get("sop_id") or data.get("sopId")
    runtime_key = data.get("runtime_key") or data.get("runtimeKey")
    code_files = data.get("code_files") or data.get("codeFiles")
    sop_steps = data.get("sop_steps") or data.get("sopSteps")

    # If not provided, try to fetch from stores
    if (not code_files or not isinstance(code_files, list)) and sop_id and sop_id in CODE_STORE:
        code_files = CODE_STORE[sop_id].get("code_files")

    if not runtime_key and sop_id and sop_id in CODE_STORE:
        runtime_key = CODE_STORE[sop_id].get("runtime_key")

    if (not sop_steps or not isinstance(sop_steps, list)) and sop_id and sop_id in SOP_STORE:
        sop_steps = SOP_STORE[sop_id].get("steps")

    # Final validation (lenient -> return 400 with clear message, not 422)
    if not runtime_key:
        raise HTTPException(status_code=400, detail="runtime_key missing and could not infer from sop_id")
    if not code_files or not isinstance(code_files, list):
        raise HTTPException(status_code=400, detail="code_files missing and could not infer from sop_id")

    explanation = grok_explain_code_text(runtime_key, code_files, sop_steps)

    # Cache explanation
    if sop_id and sop_id in CODE_STORE:
        CODE_STORE[sop_id]["explanation_text"] = explanation

    return {"explanation": explanation}

# --------------------------------------------------------------------------------------
# Root (optional 404)
# --------------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Workbench Studio Backend. See /docs for OpenAPI."}
