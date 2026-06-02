"""Local web UI server for the Automated Reward Repairer pipeline.

Bridges between the running pipeline (the /repair-* skills, executed by
Claude in a terminal session) and the user's browser. The pipeline calls
helper scripts in `web_ui/cli/` which POST messages and questions into
this server; the server pushes them to the connected browser over a
Server-Sent Events (SSE) stream and accepts the user's answers back.

Run with:
    python -m web_ui.server  [--host 127.0.0.1] [--port 8000]

Endpoints
---------
GET  /                         → serves the SPA (`static/index.html`)
GET  /static/<path>            → static assets
GET  /workspace/<path>         → files under <repo>/workspaces/ (viz mp4/png)
GET  /events                   → SSE: live message stream + initial replay
GET  /api/state                → current workspace + minimal status
POST /api/state                → set workspace / status (called by CLI)
GET  /api/messages             → all messages so far (browser fallback)
POST /api/message              → append a message (called by post_msg.py)
POST /api/question             → append a question, returns question id
POST /api/answer/<qid>         → submit an answer (called by browser)
GET  /api/wait/<qid>           → long-poll for an answer (called by ask.py)
POST /api/clear                → clear feed (new run)

Message schema (`POST /api/message` and `/api/question` body):
    {
      "type": "text" | "markdown" | "code" | "image" | "video"
              | "comparison" | "question" | "divider" | "status",
      "content": ...,
      ...type-specific fields...
    }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import uvicorn


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
WORKSPACES_DIR = REPO_ROOT / "workspaces"

app = FastAPI(title="Automated Reward Repairer — Web UI")


class _State:
    """In-process mutable state. One run at a time; in-memory is fine."""

    def __init__(self):
        self.messages: list[dict] = []
        self.pending_questions: dict[str, dict] = {}
        self.answers: dict[str, dict] = {}
        self.workspace: str | None = None
        self.status: str = "idle"
        self.subscribers: list[asyncio.Queue] = []
        self.answer_events: dict[str, asyncio.Event] = defaultdict(asyncio.Event)


S = _State()


async def _broadcast(event: dict) -> None:
    for q in list(S.subscribers):
        try:
            await q.put(event)
        except Exception:
            pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# -------- message / question ingestion (called by the CLI helpers) ---------

@app.post("/api/message")
async def post_message(req: Request) -> dict:
    body = await req.json()
    msg = {
        "id": str(uuid.uuid4()),
        "ts": _now(),
        "type": body.get("type", "text"),
        **{k: v for k, v in body.items() if k != "type"},
    }
    S.messages.append(msg)
    await _broadcast({"event": "message", "data": msg})
    return {"id": msg["id"]}


@app.post("/api/question")
async def post_question(req: Request) -> dict:
    body = await req.json()
    qid = str(uuid.uuid4())
    q = {
        "id": qid,
        "ts": _now(),
        "type": "question",
        "answered": False,
        **{k: v for k, v in body.items() if k not in ("id", "type")},
    }
    S.pending_questions[qid] = q
    S.messages.append(q)
    await _broadcast({"event": "message", "data": q})
    return {"id": qid}


@app.post("/api/cta")
async def post_cta(req: Request) -> dict:
    """Call-to-action button (single-click 'continue' UI), tracked like a question."""
    body = await req.json()
    qid = str(uuid.uuid4())
    q = {
        "id": qid,
        "ts": _now(),
        "type": "cta",
        "answered": False,
        **{k: v for k, v in body.items() if k not in ("id", "type")},
    }
    S.pending_questions[qid] = q
    S.messages.append(q)
    await _broadcast({"event": "message", "data": q})
    return {"id": qid}


@app.post("/api/answer/{qid}")
async def post_answer(qid: str, req: Request) -> dict:
    if qid not in S.pending_questions:
        raise HTTPException(404, "Unknown question id")
    body = await req.json()
    answer = {"value": body.get("value"), "notes": body.get("notes")}
    S.answers[qid] = answer
    # mark question as answered in the feed
    for m in S.messages:
        if m.get("id") == qid:
            m["answered"] = True
            m["answer"] = answer
            await _broadcast({"event": "answered", "qid": qid, "answer": answer})
            break
    S.pending_questions.pop(qid, None)
    S.answer_events[qid].set()
    return {"ok": True}


@app.get("/api/wait/{qid}")
async def wait_answer(qid: str, timeout: float = 30.0) -> dict:
    """Long-poll for an answer. Times out (408) so the caller can re-poll."""
    if qid in S.answers:
        return S.answers[qid]
    ev = S.answer_events[qid]
    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        raise HTTPException(408, "no answer yet")
    return S.answers[qid]


# -------- state -------------------------------------------------------------

@app.get("/api/state")
async def get_state() -> dict:
    return {"workspace": S.workspace, "status": S.status}


@app.post("/api/state")
async def set_state(req: Request) -> dict:
    body = await req.json()
    if "workspace" in body:
        S.workspace = body["workspace"]
    if "status" in body:
        S.status = body["status"]
    await _broadcast({"event": "state", "data": {
        "workspace": S.workspace, "status": S.status,
    }})
    return {"ok": True}


@app.post("/api/clear")
async def clear_state() -> dict:
    S.messages.clear()
    S.pending_questions.clear()
    S.answers.clear()
    S.answer_events.clear()
    await _broadcast({"event": "clear"})
    return {"ok": True}


@app.get("/api/messages")
async def get_messages() -> list[dict]:
    return S.messages


# -------- SSE stream --------------------------------------------------------

@app.get("/events")
async def sse_endpoint() -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    S.subscribers.append(queue)

    async def gen():
        try:
            init = {
                "event": "init",
                "state": {"workspace": S.workspace, "status": S.status},
                "messages": S.messages,
            }
            yield f"data: {json.dumps(init)}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # SSE comment, keeps the conn open
        finally:
            if queue in S.subscribers:
                S.subscribers.remove(queue)

    return StreamingResponse(gen(), media_type="text/event-stream")


# -------- workspace files (used by the browser to load viz mp4/png) --------

@app.get("/workspace/{path:path}")
async def serve_workspace(path: str):
    full = (WORKSPACES_DIR / path).resolve()
    try:
        full.relative_to(WORKSPACES_DIR.resolve())
    except ValueError:
        raise HTTPException(403, "path escapes workspaces/")
    if not full.exists() or not full.is_file():
        raise HTTPException(404)
    return FileResponse(full)


# -------- frontend ---------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
