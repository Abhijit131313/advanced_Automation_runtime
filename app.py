import os
import json
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq import Groq

# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Workbench Studio Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to Bolt frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global In-Memory Store
# -----------------------------------------------------------------------------
SOP_STORE: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Grok Client
# -----------------------------------------------------------------------------
def get_grok_client():
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise RuntimeError("GROK_API_KEY not configured in environment")
    return Groq(api_key=api_key)

# -----------------------------------------------------------------------------
# Helper Generators (Fixed format strings)
# -----------------------------------------------------------------------------
def generate_bpmn_xml(title: str, steps: List[str]) -> str:
    process_id = title.lower().replace(" ", "_")
    tasks = "\n".join(
        [f'<bpmn:task id="Task_{i}" name="{step}"/>' for i, step in enumerate(steps, start=1)]
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" id="Definitions_{process_id}">
  <bpmn:process id="{process_id}" isExecutable="true">
    {tasks}
  </bpmn:process>
</bpmn:definitions>
"""

def generate_camel_java_dsl(title: str, steps: List[str]) -> str:
    class_name = title.title().replace(" ", "")
    route_steps = "\n".join(
        [f'                .to("direct:{step.lower().replace(" ", "_")}")' for step in steps]
    )
    return (
        "import org.apache.camel.builder.RouteBuilder;\n\n"
        f"public class {class_name}Route extends RouteBuilder {{\n"
        "    @Override\n"
        "    public void configure() throws Exception {\n"
        f'        from("direct:start")\n'
        f"{route_steps};\n"
        "    }\n"
        "}\n"
    )

def generate_temporal_java(title: str, steps: List[str]) -> str:
    workflow_interface = f"IWorkflow{uuid.uuid4().hex[:6]}"
    workflow_impl = f"WorkflowImpl{uuid.uuid4().hex[:6]}"

    activity_calls = "\n        ".join(
        [f"activities.executeStep{i}();" for i in range(1, len(steps) + 1)]
    )
    activities = "\n".join(
        [f"// Activity {i}: {step}" for i, step in enumerate(steps, start=1)]
    )

    return (
        f"package com.example.temporal;\n\n"
        "import io.temporal.workflow.Workflow;\n\n"
        f"public interface {workflow_interface} {{\n"
        "    void execute();\n"
        "}\n\n"
        "/* Implementation */\n"
        f"public class {workflow_impl} implements {workflow_interface} {{\n"
        "    private final Activities activities = Workflow.newActivityStub(Activities.class);\n\n"
        "    @Override\n"
        "    public void execute() {\n"
        f"        {activity_calls}\n"
        "    }\n"
        "}\n\n"
        "/* Activities interface (skeleton) */\n"
        "interface Activities {\n"
        f"{activities}\n"
        "}\n"
    )

def generate_knative_yaml(title: str, steps: List[str]) -> str:
    services = "\n".join(
        [
            f"- apiVersion: serving.knative.dev/v1\n  kind: Service\n  metadata:\n"
            f"    name: {step.lower().replace(' ', '-')}\n  spec:\n"
            f"    template:\n      spec:\n        containers:\n"
            f"        - image: example/{step.lower().replace(' ', '-')}:latest"
            for step in steps
        ]
    )
    return f"---\n{services}\n"

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class GenerateCodeRequest(BaseModel):
    sop_id: str
    runtime_key: str
    options: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/sop/upload")
async def upload_sop(
    file: UploadFile = File(...),
    text: str = Form(""),
    scenario_id: str = Form(""),
):
    """Upload SOP file (pdf/txt) or plain text."""
    sop_id = str(uuid.uuid4())
    sop_text = text or (await file.read()).decode("utf-8", errors="ignore")

    # Very simple heuristic (can be replaced by Grok later)
    steps = [line.strip() for line in sop_text.split("\n") if line.strip()]
    automated = [s for i, s in enumerate(steps) if i % 2 == 0]
    manual = [s for i, s in enumerate(steps) if i % 2 != 0]

    SOP_STORE[sop_id] = {
        "text": sop_text,
        "steps": steps,
        "automated": automated,
        "manual": manual,
    }

    suggested_runtimes = [
        {"key": "temporal", "name": "TEMPORAL", "confidence": 0.55,
         "reason": "Temporal (Java/TS) - good for code-first durable workflows"},
        {"key": "bpmn", "name": "BPMN", "confidence": 0.5,
         "reason": "BPMN (XML) - good for human tasks & manual approvals"},
        {"key": "camel", "name": "CAMEL", "confidence": 0.5,
         "reason": "Apache Camel (Java DSL) - good for API/adapter integration"},
    ]

    return {
        "sop_id": sop_id,
        "summary": f"Detected {len(steps)} steps. Will automate {len(automated)}; {len(manual)} remain manual.",
        "automated_actions": automated,
        "manual_actions": manual,
        "suggested_runtimes": suggested_runtimes,
    }

@app.post("/api/generate_code")
async def generate_code(req: GenerateCodeRequest):
    sop = SOP_STORE.get(req.sop_id)
    if not sop:
        raise HTTPException(status_code=404, detail="sop_id not found")

    title = "Workflow"
    steps = sop["steps"]

    if req.runtime_key == "bpmn":
        content = generate_bpmn_xml(title, steps)
        editor = "xml"
    elif req.runtime_key == "camel":
        content = generate_camel_java_dsl(title, steps)
        editor = "java"
    elif req.runtime_key == "temporal":
        content = generate_temporal_java(title, steps)
        editor = "java"
    elif req.runtime_key == "knative":
        content = generate_knative_yaml(title, steps)
        editor = "yaml"
    else:
        raise HTTPException(status_code=400, detail="Unsupported runtime")

    SOP_STORE[req.sop_id]["last_code"] = content

    return {
        "code_files": [{"path": f"main.{editor}", "content": content}],
        "main_file": f"main.{editor}",
        "editor_mode": editor,
        "ui_schema": {},
        "confidence": 0.55,
    }

@app.post("/api/run_test")
async def run_test(payload: dict = Body(...)):
    sop_id = payload.get("sop_id")
    if not sop_id or sop_id not in SOP_STORE:
        raise HTTPException(status_code=404, detail="sop_id not found")

    return {"status": "success", "log": f"Ran workflow {sop_id} with mock test."}

@app.post("/api/chat_agent")
async def chat_agent(payload: dict = Body(...)):
    sop_id = payload.get("sop_id")
    message = payload.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    return {"reply": f"Agentic reply to: {message}"}

@app.post("/api/chat_agent_suggest_edit")
async def chat_agent_suggest_edit(payload: dict = Body(...)):
    return {"suggestion": "Consider renaming activity identifiers for clarity."}

@app.post("/api/apply_code_edit")
async def apply_code_edit(payload: dict = Body(...)):
    return {"status": "ok", "new_code": payload.get("new_code", "")}

@app.post("/api/visualize_code")
async def visualize_code(payload: dict = Body(...)):
    sop_id = payload.get("sop_id")
    runtime = payload.get("runtime_key")
    if not sop_id or sop_id not in SOP_STORE:
        raise HTTPException(status_code=404, detail="sop_id not found")
    last_code = SOP_STORE[sop_id].get("last_code", "")
    return {
        "nodes": [{"id": f"step{i}", "label": step} for i, step in enumerate(SOP_STORE[sop_id]["steps"], 1)],
        "edges": [{"from": f"step{i}", "to": f"step{i+1}"} for i in range(len(SOP_STORE[sop_id]["steps"]) - 1)],
        "raw_code": last_code,
    }

# -----------------------------------------------------------------------------
# NEW ENDPOINT: Explain Code
# -----------------------------------------------------------------------------
@app.post("/api/explain_code")
async def explain_code(payload: dict = Body(...)):
    """
    Generate a narrative explanation of the workflow automation code.
    """
    sop_id = payload.get("sop_id")
    runtime_key = payload.get("runtime_key")
    code = payload.get("code")

    if not code:
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    client = get_grok_client()

    system_prompt = f"""
You are an expert workflow automation analyst. 
Explain the given workflow automation code clearly and narratively.

Runtime: {runtime_key}

Instructions:
1. Identify what the workflow is about (e.g., Temporal workflow for dispute resolution).
2. Describe its structure (workflow interface, implementation, DSL definition, etc.).
3. Explain what each step/activity/task does in plain English.
4. Conclude with a summary of how this workflow orchestrates the process.
"""

    user_prompt = f"Here is the code:\n\n{code}"

    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        explanation = response.choices[0].message["content"]
        return {"sop_id": sop_id, "runtime_key": runtime_key, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain code failed: {str(e)}")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
