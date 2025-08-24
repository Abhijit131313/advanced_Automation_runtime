# Workbench Studio Backend

FastAPI backend for Workbench Studio (SOP → automation). Features:

- **LLM-first** SOP analysis: summary + automated vs manual steps (fallback to heuristic).
- **LLM-first** top-3 runtime suggestions with confidences (fallback to heuristic).
- Code generation: **BPMN XML**, **Apache Camel (Java DSL)**, **Knative YAML**, **Temporal Java**.
- Agentic editor:
  - `/api/chat_agent` (contextual chat over SOP)
  - `/api/chat_agent_suggest_edit` (asks LLM for concrete edit `{path,new_content,explanation}`)
  - `/api/apply_code_edit` (applies edit and versions it)
- Visualization: `/api/visualize_code` → nodes/edges diagram JSON for UI.
- Test simulator: `/api/run_test`.

## Endpoints (JSON unless noted)

- `POST /api/sop/upload` (multipart or form)
  - Fields: `file` (optional) OR `text`, optional `scenario_id`
  - Returns: `sop_id`, `summary`, `automated_actions[]`, `manual_actions[]`, `suggested_runtimes[]`

- `POST /api/generate_code`
  - Body: `{ "sop_id": "...", "runtime_key": "bpmn|camel|knative|temporal", "options": {} }`
  - Returns: `code_files[]`, `main_file`, `editor_mode`, `ui_schema`, `confidence`

- `POST /api/run_test`
  - Body: `{ "sop_id": "...", "runtime_key": "...", "test_input": {...} }`

- `POST /api/chat_agent`
  - Body: `{ "sop_id": "...", "message": "..." }`
  - Returns: `{ "reply": "...", "meta": {...} }`

- `POST /api/chat_agent_suggest_edit`
  - Body: `{ "sop_id":"...", "message":"...", "target_file":"(optional)" }`
  - Returns: `{ "reply":"...", "suggestion": { "path":"...", "new_content":"...", "explanation":"..." } }`

- `POST /api/apply_code_edit`
  - Body: `{ "sop_id":"...", "path":"...", "new_content":"...", "author":"agent|user" }`
  - Returns: updated `code_files[]` and `version_id`.

- `POST /api/visualize_code` (form)
  - Fields: `sop_id`, `runtime_key`
  - Returns: `{ nodes:[], edges:[], meta:{} }`

- `GET /api/health`

## Deploy (Render)
1. New **Python** web service.
2. Build: `pip install -r requirements.txt`
3. Start: provided by `Procfile`.
4. Env vars:
   - `GROK_API_URL` (e.g., your Grok completions/chat endpoint)
   - `GROK_API_KEY`
   - `DEFAULT_MODEL` (optional, default `gpt-5`)
5. Deploy.

## Local dev

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GROK_API_URL="https://<your-grok-endpoint>"
export GROK_API_KEY="<your-key>"
uvicorn app:app --reload --host 0.0.0.0 --port 8000
