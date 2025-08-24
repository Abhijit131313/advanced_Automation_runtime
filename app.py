import os
import io
import uuid
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from string import Template

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

# Optional Grok (Groq) import — only used if key is provided
try:
    from groq import Groq
except Exception:
    Groq = None  # type: ignore

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workbench-backend")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Workbench Studio Backend", version="0.1.0")

# CORS (allow Bolt and localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# In-memory stores (simple demo; replace with DB in production)
# -----------------------------------------------------------------------------
SOPS: Dict[str, Dict[str, Any]] = {}
# SOPS[sop_id] = {
#   "title": str,
#   "text": str,
#   "steps": List[str],
# }

SESSIONS: Dict[str, Dict[str, Any]] = {}
# SESSIONS[sop_id] = {
#   "runtime_key": str,
#   "code_files": List[{"path": str, "content": str}],
#   "main_file": str,
#   "editor_mode": str,
#   "explanation": Optional[str],
# }

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_sop_text(text: str) -> Tuple[str, List[str]]:
    """
    Parse SOP text into (title, steps).
    First non-empty line becomes title. Remaining non-empty lines become steps.
    """
    if not text:
        return ("Untitled Scenario", [])
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empties
    if not lines:
        return ("Untitled Scenario", [])
    title = lines[0]
    # Treat remaining lines that look like bullets or sentences as steps
    bullets = []
    for ln in lines[1:]:
        # strip bullet markers
        cleaned = ln
        for pfx in ["-", "*", "•", "1.", "2.", "3."]:
            if cleaned.startswith(pfx):
                cleaned = cleaned[len(pfx):].strip()
        if cleaned:
            bullets.append(cleaned)
    if not bullets:  # If no bullets: use remaining lines
        bullets = lines[1:]
    return (title, bullets)


def suggested_runtimes_for_steps(steps: List[str]) -> List[Dict[str, Any]]:
    """
    Return 3 runtime suggestions with confidence & reasons.
    Very simple heuristic for demo.
    """
    n = len(steps)
    out = [
        {"key": "temporal", "name": "TEMPORAL", "confidence": 0.55, "reason": "Temporal (Java/TS) - code-first durable workflows"},
        {"key": "bpmn", "name": "BPMN", "confidence": 0.50, "reason": "BPMN (XML) - human tasks & approvals"},
        {"key": "camel", "name": "CAMEL", "confidence": 0.50, "reason": "Apache Camel (Java DSL) - routing, adapters, APIs"},
    ]
    if n <= 2:
        out[1]["confidence"] = 0.52
        out[2]["confidence"] = 0.48
    elif n >= 8:
        out[0]["confidence"] = 0.60
    return out


def render_template(tpl: str, **kwargs) -> str:
    """Use $placeholders to avoid Python brace conflicts in Java/XML/YAML."""
    return Template(tpl).safe_substitute(**kwargs)


def generate_bpmn_xml(title: str, steps: List[str]) -> str:
    """
    Generates a minimal BPMN XML diagram with sequential tasks.
    """
    tasks_xml = []
    seq_flows = []
    last_id = "startEvent_1"
    for idx, step in enumerate(steps, start=1):
        task_id = f"Task_{idx}"
        tasks_xml.append(
            f'<bpmn:task id="{task_id}" name="{step}"/>'
        )
        seq_id = f"Flow_{idx}"
        seq_flows.append(
            f'<bpmn:sequenceFlow id="{seq_id}" sourceRef="{last_id}" targetRef="{task_id}"/>'
        )
        last_id = task_id

    # end event
    tasks_xml.append('<bpmn:endEvent id="endEvent_1" name="End"/>')
    seq_flows.append(f'<bpmn:sequenceFlow id="Flow_end" sourceRef="{last_id}" targetRef="endEvent_1"/>')

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                  xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL"
                  id="Definitions_1" targetNamespace="http://example.com/bpmn">
  <bpmn:process id="Process_1" name="{title}" isExecutable="true">
    <bpmn:startEvent id="startEvent_1" name="Start"/>
    {''.join(tasks_xml)}
    {''.join(seq_flows)}
  </bpmn:process>
</bpmn:definitions>
"""
    return xml


def generate_camel_java_dsl(title: str, steps: List[str]) -> str:
    """
    Generates a simple Camel RouteBuilder class in Java DSL.
    """
    route_steps = []
    for idx, step in enumerate(steps, start=1):
        # Represent each step as a log statement; real impl would add processors/components
        route_steps.append(f'            .process(exchange -> System.out.println("Step {idx}: {step}"))\n')

    tpl = """package com.example.camel;

import org.apache.camel.builder.RouteBuilder;

public class ${CLS} extends RouteBuilder {
    @Override
    public void configure() throws Exception {
        from("timer:${lower}?period=60000")
            .routeId("${CLS}Route")
${steps}            .to("log:${lower}?level=INFO");
    }
}
"""
    class_name = "Route" + uuid.uuid4().hex[:8]
    return render_template(
        tpl,
        CLS=class_name,
        lower=class_name.lower(),
        steps="".join(route_steps),
    )


def generate_temporal_java(title: str, steps: List[str]) -> str:
    """
    Generates a minimal Temporal workflow & activities (Java).
    """
    activity_comments = []
    activity_calls = []
    for idx, step in enumerate(steps, start=1):
        activity_comments.append(f"    // Activity {idx}: {step}\n")
        activity_calls.append(f"        activities.executeStep{idx}();\n")

    workflow_interface = "IWorkflow" + uuid.uuid4().hex[:6]
    workflow_impl = "WorkflowImpl" + uuid.uuid4().hex[:6]

    tpl = """package com.example.temporal;

import io.temporal.workflow.Workflow;

public interface ${IFACE} {
    void execute();
}

/* Implementation */
public class ${IMPL} implements ${IFACE} {
    private final Activities activities = Workflow.newActivityStub(Activities.class);

    @Override
    public void execute() {
${calls}    }
}

/* Activities interface (skeleton) */
interface Activities {
${acts}}
"""
    return render_template(
        tpl,
        IFACE=workflow_interface,
        IMPL=workflow_impl,
        calls="".join(activity_calls),
        acts="".join(activity_comments),
    )


def generate_knative_yaml(title: str, steps: List[str]) -> str:
    """
    Generates a simplistic Knative-like YAML (pseudo for demo).
    """
    tasks_yaml = []
    for idx, step in enumerate(steps, start=1):
        tasks_yaml.append(f"  - name: step{idx}\n    image: example/worker:latest\n    args: [\"{step}\"]\n")
    tpl = """apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ${name}
spec:
  template:
    spec:
      containers:
${tasks}
"""
    name = "svc-" + uuid.uuid4().hex[:6]
    return render_template(tpl, name=name, tasks="".join(tasks_yaml))


def detect_editor_mode(runtime_key: str) -> str:
    rk = (runtime_key or "").lower()
    if rk == "bpmn":
        return "xml"
    if rk == "camel":
        return "java"
    if rk == "knative":
        return "yaml"
    if rk == "temporal":
        return "java"
    return "text"


def grok_client() -> Optional[Any]:
    """
    Return a Grok (Groq) client or None if unavailable or no key present.
    NOTE: Do NOT pass 'proxies' — the SDK doesn't accept it (fix for your earlier error).
    """
    key = os.getenv("GROK_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key or Groq is None:
        logger.info("explain_code_text: no GROK_API_KEY or groq lib missing; will use heuristic")
        return None
    try:
        client = Groq(api_key=key)  # no proxies!
        return client
    except Exception as e:
        logger.info(f"explain_code_text: grok client init failed ({e}); will use heuristic")
        return None


def explain_with_grok(runtime_key: Optional[str], editor_mode: Optional[str], main_file_content: str) -> Optional[str]:
    client = grok_client()
    if client is None:
        return None
    try:
        prompt = (
            "Explain in clear non-technical language what the following automation workflow does, "
            "step by step, referencing business intent. Keep it concise but comprehensive.\n\n"
            f"Runtime: {runtime_key or 'unknown'}\n"
            f"Editor Mode: {editor_mode or 'unknown'}\n"
            "Main file content:\n"
            "----------------------------------------\n"
            f"{main_file_content[:4000]}\n"
            "----------------------------------------\n"
        )
        resp = client.chat.completions.create(
            model=os.getenv("GROK_MODEL", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": "You explain automation workflows for process designers."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content.strip()
        logger.info("explain_code_text: grok success")
        return txt
    except Exception as e:
        logger.info(f"explain_code_text: grok failed ({e}); falling back to heuristic")
        return None


def heuristic_explanation(runtime_key: Optional[str], editor_mode: Optional[str], main_file: Optional[str], code_files: List[Dict[str, str]]) -> str:
    bullets = []
    if runtime_key:
        bullets.append(f"Runtime: {runtime_key.upper()}")
    if editor_mode:
        bullets.append(f"Editor: {editor_mode}")
    if main_file:
        bullets.append(f"Main file: {main_file}")
    try:
        bullets.append("Files in bundle: " + ", ".join([cf["path"] for cf in code_files]))
    except Exception:
        pass
    return (
        "This automation defines a workflow and executes its steps in order. "
        "It orchestrates the business process by invoking each activity/task and ensuring state transitions.\n\n"
        + "\n".join(f"- {b}" for b in bullets)
        + "\n\nNote: A detailed explanation could not be generated via Grok; this is a heuristic summary."
    )

# -----------------------------------------------------------------------------
# Schemas
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


class VisualizeRequest(BaseModel):
    sop_id: Optional[str] = None
    runtime_key: Optional[str] = None
    code_files: Optional[List[CodeFile]] = None
    main_file: Optional[str] = None


class RunTestRequest(BaseModel):
    sop_id: Optional[str] = None
    runtime_key: Optional[str] = None
    code_files: Optional[List[CodeFile]] = None
    main_file: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    sop_id: Optional[str] = None
    message: str


class SuggestEditRequest(BaseModel):
    sop_id: Optional[str] = None
    instruction: str


class ApplyCodeEditRequest(BaseModel):
    sop_id: str
    file_path: str
    new_content: str


class ExplainCodeRequest(BaseModel):
    sop_id: Optional[str] = None
    runtime_key: Optional[str] = None
    code_files: Optional[List[CodeFile]] = None
    main_file: Optional[str] = None
    editor_mode: Optional[str] = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/sop/upload", response_model=UploadSOPResponse)
async def upload_sop(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    scenario_id: Optional[str] = Form(None),
):
    """
    Accept SOP via file (pdf/txt) or plain text. Extract title + steps and store.
    """
    sop_text = text or ""
    if (not sop_text) and file:
        # Very simple text extraction for demo — in real code, parse PDF properly
        raw = await file.read()
        try:
            sop_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            sop_text = "<binary or non-utf8 document>"

    title, steps = parse_sop_text(sop_text)
    sop_id = str(uuid.uuid4())
    SOPS[sop_id] = {"title": title, "text": sop_text, "steps": steps}

    auto_cnt = max(0, len(steps) - 0)  # demo: pretend all steps can be automated
    man_cnt = 0
    summary = f"Detected {len(steps)} steps. Will automate {auto_cnt}; {man_cnt} remain manual."

    logger.info(f"Uploaded SOP {sop_id} title={title} steps={len(steps)}")

    return {
        "sop_id": sop_id,
        "summary": summary,
        "automated_actions": steps or ["<binary or non-utf8 document>"],
        "manual_actions": [],
        "suggested_runtimes": suggested_runtimes_for_steps(steps),
    }


@app.post("/api/generate_code", response_model=GenerateCodeResponse)
async def generate_code(req: GenerateCodeRequest):
    sop = SOPS.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")

    title = sop.get("title") or "Untitled"
    steps = sop.get("steps") or []
    runtime_key = (req.runtime_key or "").lower().strip()

    code_files: List[Dict[str, str]] = []
    main_file = ""
    editor_mode = detect_editor_mode(runtime_key)
    ui_schema: Dict[str, Any] = {}

    if runtime_key == "bpmn":
        content = generate_bpmn_xml(title, steps)
        code_files = [{"path": "workflow.bpmn", "content": content}]
        main_file = "workflow.bpmn"
        ui_schema = {"diagram": "bpmn"}
        confidence = 0.55
    elif runtime_key == "camel":
        content = generate_camel_java_dsl(title, steps)
        code_files = [{"path": "RouteBuilder.java", "content": content}]
        main_file = "RouteBuilder.java"
        ui_schema = {"ide": "java"}
        confidence = 0.52
    elif runtime_key == "knative":
        content = generate_knative_yaml(title, steps)
        code_files = [{"path": "service.yaml", "content": content}]
        main_file = "service.yaml"
        ui_schema = {"ide": "yaml"}
        confidence = 0.50
    elif runtime_key == "temporal":
        content = generate_temporal_java(title, steps)
        code_files = [{"path": "Workflow.java", "content": content}]
        main_file = "Workflow.java"
        ui_schema = {"ide": "java"}
        confidence = 0.58
    else:
        # Fallback to a plain text pseudo-code
        lines = ["# Pseudo-workflow"]
        for i, s in enumerate(steps, 1):
            lines.append(f"{i}. {s}")
        content = "\n".join(lines)
        code_files = [{"path": "workflow.txt", "content": content}]
        main_file = "workflow.txt"
        editor_mode = "text"
        ui_schema = {"ide": "text"}
        confidence = 0.45

    # Persist in session for later hydration
    SESSIONS[req.sop_id] = {
        "runtime_key": runtime_key,
        "code_files": code_files,
        "main_file": main_file,
        "editor_mode": editor_mode,
    }

    # Optionally pre-generate explanation (not required)
    main_content = code_files[0]["content"] if code_files else ""
    explanation = explain_with_grok(runtime_key, editor_mode, main_content)
    if explanation is None:
        explanation = heuristic_explanation(runtime_key, editor_mode, main_file, code_files)
        explain_source = "heuristic"
    else:
        explain_source = "grok"
    SESSIONS[req.sop_id]["explanation"] = explanation

    logger.info(f"Generated code for sop_id={req.sop_id} runtime={runtime_key} (explain: {explain_source})")

    return {
        "code_files": code_files,
        "main_file": main_file,
        "editor_mode": editor_mode,
        "ui_schema": ui_schema,
        "confidence": 0.55,
    }


@app.post("/api/visualize_code")
async def visualize_code(req: VisualizeRequest):
    """
    Returns a simple graph (nodes/edges) visualization derived from stored SOP or provided code.
    """
    steps: List[str] = []
    if req.sop_id and req.sop_id in SOPS:
        steps = SOPS[req.sop_id].get("steps", [])
    elif req.code_files and req.main_file:
        # Very naive extraction for demo
        main = next((cf for cf in req.code_files if cf.path == req.main_file), None)
        if main:
            # try to split lines as steps
            lines = [ln.strip() for ln in main.content.splitlines() if ln.strip()]
            steps = [ln for ln in lines[:10]]

    nodes = []
    edges = []
    prev = None
    for i, s in enumerate(steps, 1):
        nid = f"n{i}"
        nodes.append({"id": nid, "label": s})
        if prev:
            edges.append({"from": prev, "to": nid})
        prev = nid
    if not nodes:
        nodes = [{"id": "n1", "label": "No steps available"}]

    return {"nodes": nodes, "edges": edges}


@app.post("/api/run_test")
async def run_test(req: RunTestRequest):
    """
    Mock test runner. Returns a simple pass with echo logs.
    """
    result = {
        "status": "passed",
        "logs": [
            "Test runner initialized.",
            f"runtime: {req.runtime_key or 'unknown'}",
            f"main_file: {req.main_file or 'unknown'}",
            "Steps executed successfully (mock).",
        ],
        "outputs": {"example": True},
    }
    return result


@app.post("/api/chat_agent")
async def chat_agent(req: ChatRequest):
    """
    Very simple echo + tip; replace with your actual agent if needed.
    """
    tip = "Tip: You can request an edit like 'rename step 3 to Validate KYC'."
    return {"reply": f"You said: {req.message}", "tip": tip}


@app.post("/api/chat_agent_suggest_edit")
async def chat_agent_suggest_edit(req: SuggestEditRequest):
    """
    Suggest a trivial edit to the main file (demo).
    """
    sop_id = req.sop_id
    if not sop_id or sop_id not in SESSIONS:
        return {"suggestion": "No code found to edit. Please generate code first."}
    main_file = SESSIONS[sop_id]["main_file"]
    return {"file_path": main_file, "new_content": "// suggested change\n" + SESSIONS[sop_id]["code_files"][0]["content"]}


@app.post("/api/apply_code_edit")
async def apply_code_edit(req: ApplyCodeEditRequest):
    """
    Apply an edit to a file in session.
    """
    sess = SESSIONS.get(req.sop_id)
    if not sess:
        raise HTTPException(status_code=404, detail="sop_id not found in session")
    updated = False
    for cf in sess["code_files"]:
        if cf["path"] == req.file_path:
            cf["content"] = req.new_content
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="file_path not found in session")
    return {"status": "ok"}


@app.post("/api/explain_code_text")
async def explain_code_text(req: Request, body: Optional[ExplainCodeRequest] = None):
    """
    Returns a plain-language explanation of what the automation code is doing.
    - Accepts camelCase or snake_case
    - Hydrates from server session by sop_id if fields are missing
    - Always 200; includes fallback heuristic if Grok isn't available
    """
    try:
        if body is None:
            raw = {}
            try:
                raw = await req.json()
            except Exception:
                raw = {}
            # Normalize keys from camelCase to snake_case
            norm: Dict[str, Any] = {}
            for k, v in raw.items():
                if k == "runtimeKey": norm["runtime_key"] = v
                elif k == "codeFiles": norm["code_files"] = v
                elif k == "mainFile": norm["main_file"] = v
                elif k == "editorMode": norm["editor_mode"] = v
                else:
                    norm[k] = v
            body = ExplainCodeRequest(**norm)
    except Exception:
        body = ExplainCodeRequest()

    sop_id = body.sop_id
    runtime_key = (body.runtime_key or "").lower().strip() or None
    code_files = body.code_files
    main_file = body.main_file
    editor_mode = (body.editor_mode or "").lower().strip() or None

    # Hydrate from session if needed
    hydrated = False
    if (not code_files or not main_file or not editor_mode or not runtime_key) and sop_id:
        try:
            session = SESSIONS.get(sop_id)
            if session:
                if not code_files and "code_files" in session:
                    # convert dicts -> CodeFile models
                    code_files = [CodeFile(**cf) for cf in session["code_files"]]
                if not main_file and "main_file" in session:
                    main_file = session["main_file"]
                if not editor_mode and "editor_mode" in session:
                    editor_mode = session["editor_mode"]
                if not runtime_key and "runtime_key" in session:
                    runtime_key = session["runtime_key"]
                hydrated = True
        except Exception as e:
            logger.info(f"explain_code_text: hydration from session failed: {e}")

    # If still missing core data, return gentle 200 message
    if not code_files or not main_file:
        msg = (
            "I don’t have the generated code to explain yet. "
            "Please run code generation first, then try again."
        )
        if sop_id and not hydrated:
            msg += " (I also attempted to hydrate from the server session using your sop_id but couldn’t find code.)"
        return JSONResponse(
            status_code=200,
            content={
                "explanation": msg,
                "used": {
                    "sop_id": sop_id,
                    "runtime_key": runtime_key,
                    "editor_mode": editor_mode,
                    "hydrated_from_session": hydrated,
                },
            },
        )

    # Get main file content
    try:
        primary = next((f for f in code_files if f.path == main_file), None) or code_files[0]
        main_content = primary.content
    except Exception:
        main_content = ""

    # Try Grok; fallback to heuristic
    explanation = explain_with_grok(runtime_key, editor_mode, main_content)
    used_model = "grok" if explanation else "heuristic"
    if not explanation:
        # Convert CodeFile models to dicts if needed
        cf_list: List[Dict[str, str]] = []
        try:
            for f in code_files:
                if isinstance(f, CodeFile):
                    cf_list.append({"path": f.path, "content": f.content})
                else:
                    cf_list.append(f)  # type: ignore
        except Exception:
            cf_list = []
        explanation = heuristic_explanation(runtime_key, editor_mode, main_file, cf_list)

    # Store latest explanation in session if sop_id
    if sop_id:
        SESSIONS.setdefault(sop_id, {})
        SESSIONS[sop_id]["explanation"] = explanation

    return {
        "explanation": explanation,
        "used": {
            "model": used_model,
            "sop_id": sop_id,
            "runtime_key": runtime_key,
            "editor_mode": editor_mode,
            "hydrated_from_session": hydrated,
        },
    }
