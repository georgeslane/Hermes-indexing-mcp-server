"""Runs deploy/mac-server/setup-embedding-server.sh against a fake oMLX server, with stand-ins for the
macOS-only commands. The script's JavaScript JSON helper runs for real, under Node instead of osascript."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "mac-server" / "setup-embedding-server.sh"

pytestmark = pytest.mark.skipif(not (shutil.which("node") and shutil.which("curl") and shutil.which("openssl")),
                                reason="needs node, curl and openssl")

SHIMS = {
    "uname": "#!/bin/bash\necho Darwin\n",
    "scutil": "#!/bin/bash\necho Mac-mini\n",
    # osascript -l JavaScript -e SCRIPT ARGS...: the same JavaScript, under Node.
    "osascript": """#!/bin/bash
shift 2; js="$2"; shift 2
exec node -e "$js
const r = run(process.argv.slice(1)); if (r !== undefined) console.log(r);" -- "$@"
""",
}

MAIN_KEY = "main-key-123"


class FakeOMLX:
    """The parts of oMLX's admin and inference API the script uses."""

    def __init__(self, settings_file: Path, have_model: bool = False):
        self.settings_file = settings_file
        self.models = [{"id": "Qwen3-8B-4bit", "model_type": "llm", "source_repo_id": None, "pinned": False}]
        if have_model:
            self.models.append(self._gemma())
        self.sub_keys: list[str] = []
        self.tasks: list[dict] = []
        self.calls: list[str] = []
        self.restarted = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                length = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(length)) if length else {}

            def _key(self):
                return (self.headers.get("Authorization") or "").removeprefix("Bearer ")

            def _route(self, method):
                path = self.path
                outer.calls.append(f"{method} {path}")
                if path == "/health":
                    return self._send(200, {"status": "healthy"})
                if path.startswith("/v1/"):
                    if self._key() not in [MAIN_KEY, *outer.sub_keys]:
                        return self._send(401, {"detail": "bad key"})
                    if path == "/v1/models":
                        return self._send(200, {"data": [{"id": m["id"]} for m in outer.models]})
                    if path == "/v1/embeddings":
                        body = self._body()
                        if body.get("model") not in [m["id"] for m in outer.models]:
                            return self._send(404, {"detail": "no such model"})
                        return self._send(200, {"data": [{"index": 0, "embedding": [0.036] * 768}]})
                if not path.startswith("/admin/api/"):
                    return self._send(404, {"detail": "not found"})
                if self._key() != MAIN_KEY:
                    return self._send(401, {"detail": "Only the main API key works here"})
                route = path.removeprefix("/admin/api")
                body = self._body() if method in ("POST", "PUT", "DELETE") else {}
                if route == "/server-info":
                    return self._send(200, {"host": "127.0.0.1", "port": 8000, "aliases": []})
                if route == "/models":
                    return self._send(200, {"models": outer.models})
                if route == "/global-settings" and method == "POST":
                    settings = json.loads(outer.settings_file.read_text())
                    settings["server"]["host"] = body["host"]
                    outer.settings_file.write_text(json.dumps(settings))
                    return self._send(200, {"success": True})
                if route == "/server/restart":
                    outer.restarted = True
                    return self._send(202, {"status": "restarting"})
                if route == "/hf/download":
                    task = {"task_id": "t1", "repo_id": body["repo_id"], "status": "downloading", "progress": 0.0,
                            "error": "", "polls": 0}
                    outer.tasks.append(task)
                    return self._send(200, {"success": True, "task": task})
                if route == "/hf/tasks":
                    for task in outer.tasks:
                        task["polls"] += 1
                        task["progress"] = min(100.0, task["polls"] * 50.0)
                        if task["polls"] >= 2 and task["status"] == "downloading":
                            task["status"] = "completed"
                            outer.models.append(outer._gemma())
                    return self._send(200, {"tasks": outer.tasks})
                if route == "/reload":
                    return self._send(200, {"status": "ok"})
                if route == "/sub-keys":
                    if method == "POST":
                        outer.sub_keys.append(body["key"])
                    else:
                        outer.sub_keys.remove(body["key"])
                    return self._send(200, {"success": True})
                match = re.fullmatch(r"/models/([^/]+)/(settings|load)", route)
                if match:
                    model = next((m for m in outer.models if m["id"] == match.group(1)), None)
                    if model is None:
                        return self._send(404, {"detail": f"Model not found: {match.group(1)}"})
                    if match.group(2) == "settings" and "is_pinned" in body:
                        model["pinned"] = body["is_pinned"]
                    return self._send(200, {"status": "ok"})
                return self._send(404, {"detail": "not found"})

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

            def do_PUT(self):
                self._route("PUT")

            def do_DELETE(self):
                self._route("DELETE")

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @staticmethod
    def _gemma():
        return {"id": "embeddinggemma-2", "model_type": "embedding", "source_repo_id": "google/embeddinggemma-2",
                "pinned": False}

    def gemma(self):
        return next(m for m in self.models if m["id"] == "embeddinggemma-2")

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def mac(tmp_path):
    shims = tmp_path / "shims"
    shims.mkdir()
    for name, body in SHIMS.items():
        (shims / name).write_text(body)
        (shims / name).chmod(0o755)
    home = tmp_path / "home"
    (home / ".omlx").mkdir(parents=True)
    env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}", "HOME": str(home), "OMLX_POLL_SECONDS": "0.2"}
    env.pop("OMLX_API_KEY", None)
    env.pop("OMLX_PORT", None)
    return home, env


def start(home: Path, host="127.0.0.1", have_model=False, api_key=MAIN_KEY) -> FakeOMLX:
    settings = home / ".omlx/settings.json"
    settings.write_text("{}")
    server = FakeOMLX(settings, have_model=have_model)
    settings.write_text(json.dumps({"server": {"host": host, "port": server.port}, "model": {"model_dir": None},
                                    "auth": {"api_key": api_key, "secret_key": None, "sub_keys": []}}))
    return server


def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=120,
                          check=False)


def test_full_setup_then_rerun_then_uninstall(mac):
    home, env = mac
    server = start(home)
    try:
        # Loopback only: refuses until told to expose oMLX.
        refused = run(env)
        assert refused.returncode == 1 and "--expose" in refused.stderr

        done = run(env, "--expose")
        assert done.returncode == 0, done.stdout + done.stderr
        assert server.restarted
        assert json.loads((home / ".omlx/settings.json").read_text())["server"]["host"] == "0.0.0.0"
        assert "POST /admin/api/hf/download" in server.calls
        assert server.gemma()["pinned"] is True
        assert len(server.sub_keys) == 1
        saved = (home / ".config/hermes-digup/embedding-server.env").read_text()
        assert f"DIGUP_EMBED_API_KEY={server.sub_keys[0]}" in saved
        assert "DIGUP_EMBED_MODEL=embeddinggemma-2" in saved
        assert f"DIGUP_EMBED_URL=http://Mac-mini:{server.port}" in done.stdout
        assert oct((home / ".config/hermes-digup/embedding-server.env").stat().st_mode)[-3:] == "600"

        # Again: the model and the key are reused, nothing is downloaded or created.
        server.calls.clear()
        again = run(env)
        assert again.returncode == 0, again.stdout + again.stderr
        assert "POST /admin/api/hf/download" not in server.calls
        assert "POST /admin/api/sub-keys" not in server.calls
        assert len(server.sub_keys) == 1

        gone = run(env, "--uninstall")
        assert gone.returncode == 0, gone.stderr
        assert server.sub_keys == [] and server.gemma()["pinned"] is False
    finally:
        server.close()


def test_existing_model_and_network_host(mac):
    home, env = mac
    server = start(home, host="127.0.0.1,100.101.102.103", have_model=True)
    try:
        done = run(env)
        assert done.returncode == 0, done.stdout + done.stderr
        assert not server.restarted
        assert "POST /admin/api/hf/download" not in server.calls
    finally:
        server.close()


def test_needs_a_main_key(mac):
    home, env = mac
    server = start(home, host="0.0.0.0", api_key=None)
    try:
        result = run(env)
        assert result.returncode == 1 and "no main API key" in result.stderr
        wrong = run(env, "--admin-key", "nope")
        assert wrong.returncode == 1 and "refused the admin key" in wrong.stderr
    finally:
        server.close()
