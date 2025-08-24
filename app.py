# app.py
import os
import uuid
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import asyncio
import aiohttp
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Workbench Studio Backend")

# Permissive CORS for dev (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Environment config
# -------------------------------
GROK_API_URL = os.getenv("GROK_API_URL")
GROK_API_KEY = os.getenv("GROK_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5")

# In-memory store (replace with DB in production)
STORE: Dict[str, Dict[str, Any]] = {}

# -------------------------------
# Pydantic models
# -------------------------------
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
    editor_mode: str
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

# Diagram models
class DiagramNode(BaseModel):
    id: str
    label: str
    type: Optional[str] = None  # e.g., 'task', 'endpoint', 'service', 'activity'

class DiagramEdge(BaseModel):
    source: str
    target: str
    label: Optional[str] = None

class DiagramResponse(BaseModel):
    nodes: List[DiagramNode]
    edges: List[DiagramEdge]
    meta: Optional[Dict[str, Any]] = {}

# Agentic editor models
class ChatEditRequest(BaseModel):
    sop_id: str
    message: str
    target_file: Optional[str] = None  # optional hint

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

# -------------------------------
# Helpers: SOP parsing & AI calls
# -------------------------------
def extract_steps_from_text(text: str) -> List[str]:
    # Extract numbered/bulleted/short lines as steps; fallback to paragraphs
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    steps: List[str] = []
    for ln in lines:
        m1 = re.match(r'^\d+\.\s+(.*)', ln)
        if m1:
            steps.append(m1.group(1))
            continue
        m2 = re.match(r'^[-*]\s+(.*)', ln)
        if m2:
            steps.append(m2.group(1))
            continue
        if 6 < len(ln) < 300 and (ln.endswith('.') or ln.startswith('Step') or len(ln.split()) < 20):
            steps.append(ln)
    if not steps:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        steps = paragraphs[:20]
    return steps

async def call_ai_model(prompt: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    # Generic AI caller; returns provider content or 'LLM_NOT_CONFIGURED'
    if GROK_API_URL and GROK_API_KEY:
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": DEFAULT_MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK_API_URL, headers=headers, json=payload, timeout=60) as resp:
                text = await resp.text()
                try:
                    j = json.loads(text)
                    if isinstance(j, dict):
                        if "choices" in j and j["choices"]:
                            c = j["choices"][0]
                            return c.get("text") or c.get("message", {}).get("content") or str(c)
                        if "output" in j:
                            return j["output"]
                    return str(j)
                except Exception:
                    return text
    await asyncio.sleep(0.01)
    return "LLM_NOT_CONFIGURED"

def heuristic_suggest_runtimes(steps: List[str]) -> List[SuggestedRuntime]:
    # Deterministic fallback if LLM fails
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
        "temporal": "Temporal (Java/TS) - good for code-first durable workflows",
    }
    suggested = [
        SuggestedRuntime(key=k, name=k.upper(), confidence=round(min(0.99, v), 2), reason=mapping[k])
        for k, v in scores.items()
    ]
    return sorted(suggested, key=lambda s: s.confidence, reverse=True)[:3]

async def ai_suggest_runtimes(steps: List[str]) -> List[SuggestedRuntime]:
    # LLM-first runtime ranking. Returns [] on failure to trigger fallback.
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

    # Extract JSON array robustly
    start = ai_text.find("["); end = ai_text.rfind("]")
    parsed = None
    if start != -1 and end != -1 and end > start:
        try: parsed = json.loads(ai_text[start:end+1])
        except Exception: parsed = None
    if parsed is None:
        try: parsed = json.loads(ai_text)
        except Exception: parsed = None
    if not isinstance(parsed, list) or not parsed:
        return []

    out: List[SuggestedRuntime] = []
    allowed = {"bpmn", "camel", "knative", "temporal"}
    for item in parsed[:3]:
        if not isinstance(item, dict):
            continue
        k = item.get("key")
        name = item.get("name") or (k.upper() if isinstance(k, str) else "UNKNOWN")
        reason = item.get("reason", "")
        conf_raw = item.get("confidence", 0)
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        if not isinstance(k, str):
            continue
        k = k.strip().lower()
        if k not in allowed:
            continue
        out.append(SuggestedRuntime(
            key=k, name=str(name),
            confidence=round(max(0.0, min(0.99, conf)), 2),
            reason=str(reason)
        ))
    return out

async def ai_summarize_sop(steps: List[str]) -> Tuple[Optional[str], Optional[List[str]], Optional[List[str]]]:
    # LLM-first SOP summary + automated/manual lists. Returns (None, None, None) on failure.
    prompt = (
        "You are an automation analyst. Given the numbered SOP steps below, return a JSON object with keys:\n
