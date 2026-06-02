"""Command-line helpers used by the /repair-* skills to drive the web UI.

The web server (see web_ui/server.py) is launched once per pipeline run on
http://127.0.0.1:8000. These subcommands send messages and questions to it
and read the user's answers back.

Subcommands
-----------
  launch                         start the server in background, idempotent
  stop                           stop the running background server
  status                         is the server running?
  clear                          reset the feed
  set <key>=<value> [...]        set workspace=… / status=…
  post --type <t> [...]          add a message to the feed
  ask  --prompt <p> [...]        add a question, block until user answers,
                                 print the answer JSON to stdout

Examples
--------
  python -m web_ui.cli launch
  python -m web_ui.cli set workspace=pandemic status=running
  python -m web_ui.cli post --type markdown --content "Hello, world."
  python -m web_ui.cli post --type code --language python --file reward.py
  python -m web_ui.cli post --type comparison --label "PAIR 1/2" \\
                            --content-file render.txt \\
                            --viz '[{"src": "/workspace/pandemic/viz/round1/pair1/A.png", "caption": "Trajectory A"}]'
  python -m web_ui.cli ask --prompt "How do you want to specify the reward?" \\
                           --options '[{"label":"Describe it","description":"..."},{"label":"Write it","description":"..."}]'
  python -m web_ui.cli ask --prompt "Fill in the template:" --input longtext --prefill "$(cat tpl.md)"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE = os.environ.get("REPAIRER_WEB_URL", "http://127.0.0.1:8000")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PID_FILE = REPO_ROOT / "web_ui" / ".server.pid"
LOG_FILE = REPO_ROOT / "web_ui" / "server.log"


# ----- HTTP helpers ---------------------------------------------------------

def _request(method: str, path: str, body: Any | None = None, *,
             timeout: float = 5.0) -> tuple[int, dict | str]:
    url = DEFAULT_BASE.rstrip("/") + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["content-type"] = "application/json"
    req = request.Request(url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, str(e)


def _server_alive() -> bool:
    try:
        code, _ = _request("GET", "/api/state", timeout=1.5)
        return code == 200
    except Exception:
        return False


# ----- launch / stop --------------------------------------------------------

def cmd_launch(args) -> int:
    if _server_alive():
        print(f"[web_ui] already running at {DEFAULT_BASE}")
        return 0

    # Spawn server as a detached background process.
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    log_fp = open(LOG_FILE, "a", buffering=1)
    log_fp.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    proc = subprocess.Popen(
        [sys.executable, "-m", "web_ui.server",
         "--host", args.host, "--port", str(args.port)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_fp,
        stderr=log_fp,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))

    # Wait for the server to come up.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if _server_alive():
            print(f"[web_ui] running at {DEFAULT_BASE}  pid={proc.pid}")
            return 0
        time.sleep(0.25)
    print(f"[web_ui] failed to come up; check {LOG_FILE}", file=sys.stderr)
    return 1


def cmd_stop(args) -> int:
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip() or 0)
        if pid:
            try:
                os.kill(pid, 15)
                print(f"[web_ui] sent SIGTERM to pid {pid}")
            except ProcessLookupError:
                pass
        PID_FILE.unlink(missing_ok=True)
    return 0


def cmd_status(args) -> int:
    alive = _server_alive()
    print(f"alive={alive}  url={DEFAULT_BASE}")
    return 0 if alive else 1


# ----- feed control ---------------------------------------------------------

def cmd_clear(args) -> int:
    code, _ = _request("POST", "/api/clear", {})
    return 0 if code == 200 else 1


def _parse_kv(items):
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--set arg must be key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def cmd_set(args) -> int:
    kv = _parse_kv(args.kv)
    code, _ = _request("POST", "/api/state", kv)
    return 0 if code == 200 else 1


# ----- post a message -------------------------------------------------------

def _read_content(args) -> str:
    if getattr(args, "content_file", None):
        return Path(args.content_file).read_text()
    if getattr(args, "content", None) is not None:
        return args.content
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def cmd_post(args) -> int:
    msg: dict[str, Any] = {"type": args.type}
    content = _read_content(args)

    if args.type in ("text", "markdown", "status", "code"):
        msg["content"] = content
        if args.type == "code":
            msg["language"] = args.language or "python"
            if args.caption:
                msg["caption"] = args.caption
    elif args.type == "comparison":
        msg["content"] = content
        if args.label:
            msg["label"] = args.label
        if args.viz:
            msg["viz"] = json.loads(args.viz)
        if args.why:
            msg["why"] = args.why
        if args.layout:
            msg["layout"] = args.layout
    elif args.type == "feature_key":
        feats_json = args.features
        if not feats_json and getattr(args, "features_file", None):
            feats_json = Path(args.features_file).read_text()
        if not feats_json:
            raise SystemExit("--features (json) or --features-file is required for feature_key")
        msg["features"] = json.loads(feats_json)
    elif args.type in ("image", "video"):
        if not args.src:
            raise SystemExit(f"--src is required for {args.type}")
        msg["src"] = args.src
        if args.caption:
            msg["caption"] = args.caption
    elif args.type == "divider":
        pass
    else:
        raise SystemExit(f"unknown --type {args.type!r}")

    code, body = _request("POST", "/api/message", msg)
    if code != 200:
        print(f"[web_ui] post failed: {code} {body}", file=sys.stderr)
        return 1
    return 0


# ----- ask a question -------------------------------------------------------

def cmd_ask(args) -> int:
    q: dict[str, Any] = {
        "prompt": args.prompt,
    }
    if args.fields or args.fields_file:
        fields_json = args.fields or Path(args.fields_file).read_text()
        q["fields"] = json.loads(fields_json)
        q["input"] = "form"
    elif args.options:
        q["options"] = json.loads(args.options)
        q["input"] = "choice"
        if args.multi:
            q["multi"] = True
    elif args.input == "longtext":
        q["input"] = "longtext"
    elif args.input == "text":
        q["input"] = "text"
    else:
        q["input"] = "text"

    if args.placeholder:
        q["placeholder"] = args.placeholder
    if args.prefill:
        q["prefill"] = args.prefill
    if args.prefill_file:
        q["prefill"] = Path(args.prefill_file).read_text()

    # Status hint
    _request("POST", "/api/state", {"status": "awaiting"})

    code, body = _request("POST", "/api/question", q)
    if code != 200 or "id" not in body:
        print(f"[web_ui] ask failed: {code} {body}", file=sys.stderr)
        return 1
    qid = body["id"]

    print(f"[web_ui] awaiting answer in browser ({DEFAULT_BASE}) …",
          file=sys.stderr, flush=True)
    # Long-poll loop. The server times out a wait after ~30 s; we re-poll.
    deadline = time.time() + args.timeout if args.timeout > 0 else float("inf")
    while time.time() < deadline:
        code, body = _request("GET", f"/api/wait/{qid}", timeout=35.0)
        if code == 200 and isinstance(body, dict) and "value" in body:
            _request("POST", "/api/state", {"status": "running"})
            json.dump(body, sys.stdout)
            sys.stdout.write("\n")
            return 0
        if code != 408:
            print(f"[web_ui] wait failed: {code} {body}", file=sys.stderr)
            return 1
    print("[web_ui] ask timed out", file=sys.stderr)
    return 2


def cmd_cta(args) -> int:
    body = {"label": args.label, "value": args.value}
    if args.prompt:
        body["prompt"] = args.prompt

    _request("POST", "/api/state", {"status": "awaiting"})
    code, resp = _request("POST", "/api/cta", body)
    if code != 200 or "id" not in resp:
        print(f"[web_ui] cta failed: {code} {resp}", file=sys.stderr)
        return 1
    qid = resp["id"]
    print(f"[web_ui] awaiting click in browser ({DEFAULT_BASE}) …",
          file=sys.stderr, flush=True)
    deadline = time.time() + args.timeout if args.timeout > 0 else float("inf")
    while time.time() < deadline:
        code, resp = _request("GET", f"/api/wait/{qid}", timeout=35.0)
        if code == 200 and isinstance(resp, dict) and "value" in resp:
            _request("POST", "/api/state", {"status": "running"})
            json.dump(resp, sys.stdout)
            sys.stdout.write("\n")
            return 0
        if code != 408:
            print(f"[web_ui] wait failed: {code} {resp}", file=sys.stderr)
            return 1
    print("[web_ui] cta timed out", file=sys.stderr)
    return 2


# ----- argparse -------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="web_ui.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_launch = sub.add_parser("launch")
    sp_launch.add_argument("--host", default="127.0.0.1")
    sp_launch.add_argument("--port", type=int, default=8000)
    sp_launch.set_defaults(fn=cmd_launch)

    sp_stop = sub.add_parser("stop"); sp_stop.set_defaults(fn=cmd_stop)
    sp_status = sub.add_parser("status"); sp_status.set_defaults(fn=cmd_status)
    sp_clear = sub.add_parser("clear"); sp_clear.set_defaults(fn=cmd_clear)

    sp_set = sub.add_parser("set")
    sp_set.add_argument("kv", nargs="+")
    sp_set.set_defaults(fn=cmd_set)

    sp_post = sub.add_parser("post")
    sp_post.add_argument("--type", required=True,
                          choices=["text", "markdown", "status", "code",
                                   "comparison", "image", "video", "divider",
                                   "feature_key"])
    sp_post.add_argument("--content")
    sp_post.add_argument("--content-file")
    sp_post.add_argument("--language")
    sp_post.add_argument("--caption")
    sp_post.add_argument("--src")
    sp_post.add_argument("--label")
    sp_post.add_argument("--viz",
                          help='JSON list of viz items. Each item: {"src","caption",'
                               '"preferred"?: bool, "role"?: "A"|"B"|"C", '
                               '"probability"?: float}. `preferred` adds a '
                               '"Designed reward function prefers this trajectory" '
                               'badge under that item (use only on the reward-preferred '
                               'side of a pair). `role` + `probability` are read in the '
                               'lottery layout.')
    sp_post.add_argument("--layout", choices=["row", "lottery"], default=None,
                          help='Comparison layout. "row" (default) renders viz items '
                               'in one flex row. "lottery" places items with role A '
                               'and B side-by-side on top (with their probabilities) '
                               'and the role-C item below.')
    sp_post.add_argument("--why", help='"Why they disagree" interpretation line')
    sp_post.add_argument("--features", help='JSON list of {"name","description"} feature items')
    sp_post.add_argument("--features-file", help='Path to JSON file with the feature list')
    sp_post.set_defaults(fn=cmd_post)

    sp_ask = sub.add_parser("ask")
    sp_ask.add_argument("--prompt", required=True)
    sp_ask.add_argument("--options",
                         help='JSON list of {"label","description","value"} option objects')
    sp_ask.add_argument("--multi", action="store_true",
                         help="allow selecting multiple options")
    sp_ask.add_argument("--fields",
                         help='JSON list of {"name","label","description","placeholder"} '
                              'objects → render the question as a form with one textarea per field')
    sp_ask.add_argument("--fields-file",
                         help="Path to JSON file with the field list (alternative to --fields)")
    sp_ask.add_argument("--input", choices=["text", "longtext"], default="text")
    sp_ask.add_argument("--placeholder")
    sp_ask.add_argument("--prefill")
    sp_ask.add_argument("--prefill-file")
    sp_ask.add_argument("--timeout", type=float, default=0.0,
                         help="seconds to wait for an answer; 0 = forever")
    sp_ask.set_defaults(fn=cmd_ask)

    sp_cta = sub.add_parser("cta",
                             help="Post a call-to-action button (single 'Continue' button) and wait for the user to click it.")
    sp_cta.add_argument("--label", required=True,
                         help="Button text (e.g. 'Identify possible flaws')")
    sp_cta.add_argument("--value", default="continue",
                         help="Value returned in the answer JSON when clicked")
    sp_cta.add_argument("--prompt", default="",
                         help="Optional prompt text shown above the button")
    sp_cta.add_argument("--timeout", type=float, default=0.0,
                         help="seconds to wait; 0 = forever")
    sp_cta.set_defaults(fn=cmd_cta)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
