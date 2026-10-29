"""Runs deploy/pi/setup-embedding-server.sh with stand-ins for git, cmake, the model download and systemd. The
"built" llama-server is a small fake that answers /health and /v1/embeddings, started by the fake systemctl the way
systemd would start the real one, so the script's own checks (health, the test query, config.env) run for real."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "setup-embedding-server.sh"
COMMIT = "4d756bc72bf00a4aacf410ae15a2d315f3db400d"
SHA256 = "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
MAC = "http://georges-mac-server.tail27b9b1.ts.net:8000"

pytestmark = pytest.mark.skipif(not (shutil.which("curl") and shutil.which("truncate")), reason="needs curl")

FAKE_SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        self._send(200, {"status": "ok"}) if self.path == "/health" else self._send(404, {})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send(200, {"data": [{"index": 0, "embedding": [0.036] * 768}]})
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""

SHIMS = {
    "uname": "#!/bin/bash\n[ \"$1\" = -m ] && echo aarch64 || echo Linux\n",
    # clone / rev-parse / fetch / checkout / apply, with the patch's state in a file.
    "git": f"""#!/bin/bash
echo "git $*" >> "$FAKE_LOG"
args=("$@")
if [ "$1" = -C ]; then dir="$2"; args=("${{@:3}}"); fi
case "${{args[0]}}" in
  -c) for a in "$@"; do last="$a"; done; mkdir -p "$last/.git" ;;
  rev-parse) echo {COMMIT} ;;
  apply)
    if [ "${{args[1]}}" = --check ]; then [ ! -f "$dir/.patched" ]
    elif [ "${{args[1]}}" = --reverse ]; then [ -f "$dir/.patched" ]
    else touch "$dir/.patched"; fi ;;
  *) ;;
esac
""",
    "cmake": """#!/bin/bash
echo "cmake $*" >> "$FAKE_LOG"
if [ "$1" = --build ]; then
  mkdir -p "$2/bin"
  printf '#!/bin/bash\\nexec "%s" "%s" "$@"\\n' "$FAKE_PYTHON" "$FAKE_SERVER" > "$2/bin/llama-server"
  chmod +x "$2/bin/llama-server"
fi
""",
    # The model download: a sparse file of the real size (no disk used); other requests go to the real curl.
    "curl": """#!/bin/bash
if [[ "$*" == *huggingface.co* ]]; then
  echo "download $*" >> "$FAKE_LOG"
  while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done
  truncate -s 309855456 "$out"; exit 0
fi
exec /usr/bin/curl "$@"
""",
    "sha256sum": f"""#!/bin/bash
case "$1" in *.gguf|*.gguf.part) echo "{SHA256}  $1" ;; *) exec /usr/bin/sha256sum "$@" ;; esac
""",
    # systemd: `restart` starts ExecStart from the unit, `disable --now` stops it.
    "systemctl": """#!/bin/bash
echo "systemctl $*" >> "$FAKE_LOG"
unit="$HOME/.config/systemd/user/hermes-digup-embeddings.service"
stop() { [ -f "$FAKE_PIDS" ] && kill "$(cat "$FAKE_PIDS")" 2>/dev/null; rm -f "$FAKE_PIDS"; }
case "$*" in
  *restart*)
    stop
    cmd="$(sed -n 's/^ExecStart=//p' "$unit")"
    eval "nohup $cmd >/dev/null 2>&1 &"
    echo $! > "$FAKE_PIDS" ;;
  *disable*) stop ;;
esac
""",
    "loginctl": "#!/bin/bash\necho yes\n",
    "journalctl": "#!/bin/bash\necho '(log)'\n",
    "sudo": "#!/bin/bash\necho \"sudo $*\" >> \"$FAKE_LOG\"; exit 1\n",
    "hermes-digup": "#!/bin/bash\necho \"hermes-digup $*\" >> \"$FAKE_LOG\"\n",
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def pi(tmp_path):
    shims = tmp_path / "shims"
    shims.mkdir()
    for name, body in SHIMS.items():
        (shims / name).write_text(body)
        (shims / name).chmod(0o755)
    server = tmp_path / "fake_llama_server.py"
    server.write_text(FAKE_SERVER)
    home = tmp_path / "home"
    config = home / ".config/hermes-digup/config.env"
    config.parent.mkdir(parents=True)
    config.write_text(f"DIGUP_INDEX_DIR=/x\nDIGUP_EMBED_URL={MAC}\nDIGUP_EMBED_API_KEY=k\n"
                      "DIGUP_EMBED_MODEL=embeddinggemma-2-bf16\n")
    env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}", "HOME": str(home),
           "FAKE_LOG": str(tmp_path / "calls.log"), "FAKE_PIDS": str(tmp_path / "server.pid"),
           "FAKE_PYTHON": sys.executable, "FAKE_SERVER": str(server)}
    yield home, config, env, tmp_path / "calls.log"
    pid_file = tmp_path / "server.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass


def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=180,
                          check=False)


def url_setting(config: Path) -> str:
    return next(line.split("=", 1)[1] for line in config.read_text().splitlines()
                if line.startswith("DIGUP_EMBED_URL="))


def test_setup_rerun_and_uninstall(pi):
    home, config, env, log = pi
    port = str(free_port())

    first = run(env, "--port", port, "--threads", "4")
    assert first.returncode == 0, first.stdout + first.stderr
    assert "768 dimensions" in first.stdout
    # The Pi first, the Mac kept as the fallback; the rest of config.env untouched.
    assert url_setting(config) == f"http://127.0.0.1:{port},{MAC}"
    assert "DIGUP_EMBED_MODEL=embeddinggemma-2-bf16" in config.read_text()
    unit = (home / ".config/systemd/user/hermes-digup-embeddings.service").read_text()
    for flag in ("--embedding", "--pooling mean", "--ubatch-size 2048", "--host 127.0.0.1", f"--port {port}",
                 "--threads 4", "embeddinggemma-2-Q8_0.gguf"):
        assert flag in unit, flag
    calls = log.read_text()
    assert "git -C" in calls and "apply" in calls          # DigUp's patch
    assert "--target llama-server" in calls
    assert "download" in calls
    assert "hermes-digup doctor" in calls
    assert "sudo" not in calls                             # tools present, lingering already on

    # Again: nothing rebuilt or downloaded, the patch not applied twice, the URL list not doubled.
    log.write_text("")
    again = run(env, "--port", port)
    assert again.returncode == 0, again.stdout + again.stderr
    calls = log.read_text()
    assert "cmake" not in calls and "download" not in calls
    assert url_setting(config) == f"http://127.0.0.1:{port},{MAC}"

    gone = run(env, "--port", port, "--uninstall")
    assert gone.returncode == 0, gone.stderr
    assert url_setting(config) == MAC
    assert not (home / ".config/systemd/user/hermes-digup-embeddings.service").exists()


def test_no_fallback(pi):
    _, config, env, _ = pi
    port = str(free_port())
    done = run(env, "--port", port, "--no-fallback")
    assert done.returncode == 0, done.stdout + done.stderr
    assert url_setting(config) == f"http://127.0.0.1:{port}"


def test_needs_install_first(pi):
    _, config, env, _ = pi
    config.unlink()
    result = run(env)
    assert result.returncode == 1 and "install.sh first" in result.stderr
