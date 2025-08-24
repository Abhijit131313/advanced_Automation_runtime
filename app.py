import os
import re
import uuid
import logging
from typing import List, Dict, Optional, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Groq is optional; we fall back if absent or key missing
try:
    from groq import Groq  # type: ignore
    _groq_available = True
except Exception:
    _groq_available = False

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------
app = FastAPI(title="Workbench Studio Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("workbench-backend")

# -----------------------------------------------------------------------------
# In-memory store
# -----------------------------------------------------------------------------
# SOP_STORE[sop_id] = {
#   "title": str,
#   "steps": List[str],
#   "last_code": { "<runtime_key>": { "files":[{path,content}], "main":str, "mode":str } }
# }
SOP_STORE: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class SuggestedRuntime(BaseModel):
    key: str
    name: str
    confidence: float
    reason: str

class CodeFile(BaseModel):
    path: str
    content: str

class GenerateCodeRequest(BaseModel):
    sop_id: str
    runtime_key: str
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class GenerateCodeResponse(BaseModel):
    code_files: List[CodeFile]
    main_file: str
    editor_mode: str
    ui_schema: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    explanation: Optional[str] = None  # NEW

class RunTestRequest(BaseModel):
    sop_id: str
    runtime_key: str
    code_files: Optional[List[CodeFile]] = None
    input: Dict[str, Any] = Field(default_factory=dict)

class RunTestResponse(BaseModel):
    status: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)

class VisualizeRequest(BaseModel):
    sop_id: Optional[str] = None
    runtime_key: Optional[str] = None
    code_file: Optional[CodeFile] = None

class VisualizeResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class ChatAgentRequest(BaseModel):
    sop_id: Optional[str] = None
    runtime_key: Optional[str] = None
    message: str
    code: Optional[str] = None

class ChatAgentResponse(BaseModel):
    reply: str

class SuggestEditRequest(BaseModel):
    code: str
    instruction: str
    runtime_key: str

class SuggestEditResponse(BaseModel):
    suggested_patch: str

class ApplyCodeEditRequest(BaseModel):
    code: str
    patch: str

class ApplyCodeEditResponse(BaseModel):
    code: str

class ExplainCodeRequest(BaseModel):
    code: str
    runtime_key: str

class ExplainCodeResponse(BaseModel):
    explanation: str
    groq_used: bool

# -----------------------------------------------------------------------------
# SOP parsing helpers
# -----------------------------------------------------------------------------
def _extract_steps_from_text(s: str) -> List[str]:
    lines = [ln.strip(" \t\r\n-*•") for ln in s.splitlines()]
    steps = [ln for ln in lines if ln]
    return steps

# -----------------------------------------------------------------------------
# Generators (brace-safe; no .format on Java blocks)
# -----------------------------------------------------------------------------
def generate_bpmn_xml(title: str, steps: List[str]) -> str:
    tasks = "\n".join(
        '    <bpmn:task id="Task_' + str(i + 1) + '" name="' + step.replace('"', "'") + '" />'
        for i, step in enumerate(steps)
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" '
        'id="Definitions_' + uuid.uuid4().hex + '">\n'
        '  <bpmn:process id="workflow" isExecutable="true">\n'
        + tasks + "\n"
        '  </bpmn:process>\n'
        '</bpmn:definitions>\n'
    )
    return xml

def generate_camel_java_dsl(title: str, steps: List[str]) -> str:
    route_steps = "\n        ".join(
        '.to("log:step' + str(i + 1) + '") // ' + step for i, step in enumerate(steps)
    )
    class_name = "RouteBuilder_" + uuid.uuid4().hex[:6]
    direct_name = re.sub(r"[^a-z0-9_]", "_", title.lower()).strip("_")
    java = (
        "package com.example.routes;\n\n"
        "import org.apache.camel.builder.RouteBuilder;\n"
        "import org.springframework.stereotype.Component;\n\n"
        "@Component\n"
        "public class " + class_name + " extends RouteBuilder {\n"
        "    @Override\n"
        "    public void configure() throws Exception {\n"
        '        from("direct:' + direct_name + '")\n'
        "        " + route_steps + ";\n"
        "    }\n"
        "}\n"
    )
    return java

def generate_temporal_java(title: str, steps: List[str]) -> str:
    iface = "IWorkflow" + uuid.uuid4().hex[:6]
    impl = "WorkflowImpl" + uuid.uuid4().hex[:6]
    acts = "Activities"

    activity_calls = "\n        ".join("activities.executeStep" + str(i + 1) + "();" for i in range(len(steps)))
    activity_comments = "\n".join("    // Activity " + str(i + 1) + ": " + step for i, step in enumerate(steps))

    java = (
        "package com.example.temporal;\n\n"
        "import io.temporal.workflow.Workflow;\n\n"
        "public interface " + iface + " {\n"
        "    void execute();\n"
        "}\n\n"
        "public class " + impl + " implements " + iface + " {\n"
        "    private final " + acts + " activities = Workflow.newActivityStub(" + acts + ".class);\n\n"
        "    @Override\n"
        "    public void execute() {\n"
        "        " + activity_calls + "\n"
        "    }\n"
        "}\n\n"
        "interface " + acts + " {\n"
        + activity_comments + "\n"
        "}\n"
    )
    return java

def generate_knative_yaml(title: str, steps: List[str]) -> str:
    containers = ""
    for i, step in enumerate(steps, 1):
        img = re.sub(r"[^a-z0-9-]", "-", step.lower()).strip("-")
        containers += "      - name: step" + str(i) + "\n        image: example/" + img + ":latest\n"
    yaml = (
        "apiVersion: serving.knative.dev/v1\n"
        "kind: Service\n"
        "metadata:\n"
        "  name: " + re.sub(r"[^a-z0-9-]", "-", title.lower()).strip("-") + "-workflow\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        + containers
    )
    return yaml

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def visualize_bpmn(xml: str) -> VisualizeResponse:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    idx = 0
    for m in re.finditer(r'<bpmn:task[^>]*name="([^"]+)"', xml):
        name = m.group(1)
        nid = "node_" + str(idx)
        nodes.append({"id": nid, "label": name})
        if idx > 0:
            edges.append({"from": "node_" + str(idx - 1), "to": nid})
        idx += 1
    return VisualizeResponse(nodes=nodes, edges=edges)

def visualize_camel(java: str) -> VisualizeResponse:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    steps = re.findall(r'\.to\("([^"]+)"\)', java)
    for i, s in enumerate(steps):
        nid = "node_" + str(i)
        nodes.append({"id": nid, "label": s})
        if i > 0:
            edges.append({"from": "node_" + str(i - 1), "to": nid})
    return VisualizeResponse(nodes=nodes, edges=edges)

def visualize_temporal(java: str) -> VisualizeResponse:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    acts = re.findall(r'activities\.([a-zA-Z0-9_]+)\s*\(', java)
    for i, a in enumerate(acts):
        nid = "node_" + str(i)
        nodes.append({"id": nid, "label": a})
        if i > 0:
            edges.append({"from": "node_" + str(i - 1), "to": nid})
    if not nodes:
        # fallback to activity comments
        for i, m in enumerate(re.finditer(r'//\s*Activity\s*\d+:\s*(.+)', java)):
            txt = m.group(1).strip()
            nid = "node_" + str(i)
            nodes.append({"id": nid, "label": txt})
            if i > 0:
                edges.append({"from": "node_" + str(i - 1), "to": nid})
    return VisualizeResponse(nodes=nodes, edges=edges)

def visualize_knative(yaml: str) -> VisualizeResponse:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    steps = re.findall(r'^\s*-\s*name:\s*([^\s]+)', yaml, flags=re.MULTILINE)
    for i, s in enumerate(steps):
        nid = "node_" + str(i)
        nodes.append({"id": nid, "label": s})
        if i > 0:
            edges.append({"from": "node_" + str(i - 1), "to": nid})
    return VisualizeResponse(nodes=nodes, edges=edges)

# -----------------------------------------------------------------------------
# Explanation (Groq + heuristic fallback) with logging
# -----------------------------------------------------------------------------
def _heuristic_explanation(code: str, runtime_key: str) -> str:
    bullets: List[str] = []

    # Comments
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("//") or s.startswith("#") or s.startswith("--"):
            msg = s.lstrip("/#-! ").strip()
            if msg:
                bullets.append("- " + msg)

    # BPMN names
    for m in re.finditer(r'name="([^"]+)"', code):
        name = m.group(1).strip()
        if name and len(name) < 200:
            bullets.append("- " + name)

    # Camel .to steps
    for i, seg in enumerate(re.findall(r'\.to\("([^"]+)"\)', code), 1):
        bullets.append("- Step " + str(i) + ": route to `" + seg + "`")

    # Temporal activities
    for m in re.finditer(r'activities\.([a-zA-Z0-9_]+)\s*\(', code):
        bullets.append("- Calls activity `" + m.group(1) + "`")

    # Deduplicate
    seen: set = set()
    uniq: List[str] = []
    for b in bullets:
        if b not in seen:
            uniq.append(b)
            seen.add(b)
    if not uniq:
        uniq = [
            "- Orchestrates a multi-step workflow.",
            "- Executes tasks in sequence with retries/integration points.",
        ]
    title = "This is a " + runtime_key.upper() + " workflow that orchestrates a business process."
    return title + "\n\n**What it does:**\n" + "\n".join(uniq)

def explain_code_text(code: str, runtime_key: str) -> Tuple[str, bool]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "llama3-70b-8192")

    if not (_groq_available and api_key):
        logger.info("explain_code_text: heuristic (Groq unavailable or key missing)")
        return _heuristic_explanation(code, runtime_key), False

    try:
        client = Groq(api_key=api_key)
        sys_prompt = (
            "You are a senior automation engineer. Given code for a workflow (BPMN XML, Camel Java DSL, "
            "Temporal Java, or Knative YAML), produce a concise developer-facing section titled exactly:\n"
            "What is this Workflow Automation doing?\n\n"
            "Then explain briefly in 5-10 bullet points. Avoid fluff."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Runtime: " + runtime_key + "\n\nCode:\n" + code[:12000]},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        text = resp.choices[0].message.content.strip()
        logger.info("explain_code_text: Groq used (model=%s)", model)
        return text, True
    except Exception as e:
        logger.warning("explain_code_text: Groq failed, fallback heuristic. error=%s", e)
        return _heuristic_explanation(code, runtime_key), False

# -----------------------------------------------------------------------------
# API: SOP upload
# -----------------------------------------------------------------------------
@app.post("/api/sop/upload")
async def upload_sop(
    file: UploadFile = File(...),
    text: str = Form(""),
    scenario_id: str = Form(""),
):
    sop_id = str(uuid.uuid4())
    if text:
        raw = text
    else:
        raw_bytes = await file.read()
        raw = raw_bytes.decode("utf-8", errors="ignore")

    steps = _extract_steps_from_text(raw)
    title = scenario_id or (file.filename or "workflow")

    SOP_STORE[sop_id] = {"title": title, "steps": steps, "last_code": {}}

    suggested = [
        SuggestedRuntime(key="temporal", name="TEMPORAL", confidence=0.88, reason="Durable, code-first workflow orchestration"),
        SuggestedRuntime(key="bpmn", name="BPMN", confidence=0.74, reason="Great for human-in-the-loop and approvals"),
        SuggestedRuntime(key="camel", name="CAMEL", confidence=0.69, reason="Strong for integration/routing"),
    ]

    summary = "Detected " + str(len(steps)) + " steps."
    logger.info("Uploaded SOP %s title=%s steps=%d", sop_id, title, len(steps))
    return {
        "sop_id": sop_id,
        "summary": summary,
        "automated_actions": steps,
        "manual_actions": [],
        "suggested_runtimes": [s.dict() for s in suggested],
    }

# -----------------------------------------------------------------------------
# API: Generate code (+ explanation)
# -----------------------------------------------------------------------------
@app.post("/api/generate_code", response_model=GenerateCodeResponse)
async def generate_code(req: GenerateCodeRequest):
    sop = SOP_STORE.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")

    title = sop["title"]
    steps: List[str] = sop["steps"]
    runtime_key = req.runtime_key.lower()

    if runtime_key == "bpmn":
        content = generate_bpmn_xml(title, steps)
        editor_mode = "xml"; main_path = "main.xml"; confidence = 0.74
    elif runtime_key == "camel":
        content = generate_camel_java_dsl(title, steps)
        editor_mode = "java"; main_path = "RouteBuilder.java"; confidence = 0.69
    elif runtime_key == "temporal":
        content = generate_temporal_java(title, steps)
        editor_mode = "java"; main_path = "Workflow.java"; confidence = 0.88
    elif runtime_key == "knative":
        content = generate_knative_yaml(title, steps)
        editor_mode = "yaml"; main_path = "service.yaml"; confidence = 0.62
    else:
        raise HTTPException(status_code=400, detail="Unsupported runtime_key")

    # Save last code
    code_files = [CodeFile(path=main_path, content=content)]
    sop["last_code"][runtime_key] = {"files": [cf.dict() for cf in code_files], "main": main_path, "mode": editor_mode}

    # Explanation (Groq → heuristic)
    explanation_text, used_groq = explain_code_text(content, runtime_key)

    logger.info("Generated code for sop_id=%s runtime=%s (explain: %s)",
                req.sop_id, runtime_key, "groq" if used_groq else "heuristic")

    return GenerateCodeResponse(
        code_files=code_files,
        main_file=main_path,
        editor_mode=editor_mode,
        ui_schema={},         # you can extend with runtime-specific schema
        confidence=confidence,
        explanation=explanation_text,
    )

# -----------------------------------------------------------------------------
# API: Run Test (stubbed)
# -----------------------------------------------------------------------------
@app.post("/api/run_test", response_model=RunTestResponse)
async def run_test(req: RunTestRequest):
    logs: List[str] = []
    if not req.code_files:
        sop = SOP_STORE.get(req.sop_id)
        if not sop:
            raise HTTPException(status_code=404, detail="sop_id not found")
        last = sop["last_code"].get(req.runtime_key.lower())
        if not last:
            raise HTTPException(status_code=404, detail="no code generated for this runtime")
        logs.append("Loaded last generated code from server memory.")
    else:
        logs.append("Using code files provided in request.")

    # Fake metrics
    metrics = {"executed_steps": 1, "errors": 0}
    logs.append("Test executed successfully.")

    return RunTestResponse(status="ok", metrics=metrics, logs=logs)

# -----------------------------------------------------------------------------
# API: Visualize
# -----------------------------------------------------------------------------
@app.post("/api/visualize_code", response_model=VisualizeResponse)
async def visualize_code(req: VisualizeRequest):
    runtime_key = (req.runtime_key or "").lower()
    code_text: Optional[str] = None

    if req.code_file:
        code_text = req.code_file.content
        runtime_key = runtime_key or _infer_runtime_from_path(req.code_file.path)

    if not code_text:
        if not req.sop_id or not runtime_key:
            raise HTTPException(status_code=400, detail="Provide sop_id+runtime_key or code_file")
        sop = SOP_STORE.get(req.sop_id)
        if not sop:
            raise HTTPException(status_code=404, detail="sop_id not found")
        last = sop["last_code"].get(runtime_key)
        if not last:
            raise HTTPException(status_code=404, detail="no code generated for this runtime")
        code_text = ""
        for f in last["files"]:
            if f["path"] == last["main"]:
                code_text = f["content"]
                break

    if runtime_key == "bpmn":
        return visualize_bpmn(code_text)
    if runtime_key == "camel":
        return visualize_camel(code_text)
    if runtime_key == "temporal":
        return visualize_temporal(code_text)
    if runtime_key == "knative":
        return visualize_knative(code_text)

    # Try to infer from content
    if "<bpmn:definitions" in code_text:
        return visualize_bpmn(code_text)
    if "extends RouteBuilder" in code_text:
        return visualize_camel(code_text)
    if "io.temporal.workflow.Workflow" in code_text:
        return visualize_temporal(code_text)
    if "kind: Service" in code_text and "serving.knative.dev" in code_text:
        return visualize_knative(code_text)

    return VisualizeResponse(nodes=[], edges=[])

def _infer_runtime_from_path(path: str) -> str:
    p = path.lower()
    if p.endswith(".xml"):
        return "bpmn"
    if p.endswith(".java"):
        # could be camel or temporal; leave to content in main flow
        return ""
    if p.endswith(".yaml") or p.endswith(".yml"):
        return "knative"
    return ""

# -----------------------------------------------------------------------------
# API: Agent Chat (simple; Groq → heuristic)
# -----------------------------------------------------------------------------
@app.post("/api/chat_agent", response_model=ChatAgentResponse)
async def chat_agent(req: ChatAgentRequest):
    message = (req.message or "").strip()
    if not message:
        # avoid UI error "Bad data: Messages cannot be empty"
        return ChatAgentResponse(reply="Please enter a question or instruction about the code.")

    # If you want, you can route to Groq here as well; for now, a simple helpful echo:
    reply = "I received your message: " + message + "\n\n" \
            "Try asking things like:\n" \
            "- “Explain step 3.”\n" \
            "- “Convert this to a service task.”\n" \
            "- “Add error handling for API failures.”"
    return ChatAgentResponse(reply=reply)

# -----------------------------------------------------------------------------
# API: Suggest code edit (very simple heuristic patch)
# -----------------------------------------------------------------------------
@app.post("/api/chat_agent_suggest_edit", response_model=SuggestEditResponse)
async def chat_agent_suggest_edit(req: SuggestEditRequest):
    # For now, return a pseudo patch comment the IDE can display
    suggestion = "// SUGGESTION (" + req.runtime_key.upper() + "): " + req.instruction.strip()
    patch = suggestion + "\n"
    return SuggestEditResponse(suggested_patch=patch)

# -----------------------------------------------------------------------------
# API: Apply code edit (append-only demo)
# -----------------------------------------------------------------------------
@app.post("/api/apply_code_edit", response_model=ApplyCodeEditResponse)
async def apply_code_edit(req: ApplyCodeEditRequest):
    new_code = req.code + "\n" + req.patch
    return ApplyCodeEditResponse(code=new_code)

# -----------------------------------------------------------------------------
# API: Explain code (standalone)
# -----------------------------------------------------------------------------
@app.post("/api/explain_code_text", response_model=ExplainCodeResponse)
async def explain_code_endpoint(req: ExplainCodeRequest):
    text, used_groq = explain_code_text(req.code, req.runtime_key)
    return ExplainCodeResponse(explanation=text, groq_used=used_groq)

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}
