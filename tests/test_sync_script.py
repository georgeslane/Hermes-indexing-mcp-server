"""Runs deploy/mac-laptop/digup-sync.sh end to end on Linux, with stand-ins for the macOS-only commands
(stat -f, defaults, scutil, pmset) and an ssh that runs the remote side in a local "Pi" folder."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest
from conftest import HOME, build_sample

from hermes_digup.config import Settings
from hermes_digup.index import IndexManager
from hermes_digup.service import DigUpService

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "mac-laptop" / "digup-sync.sh"

pytestmark = pytest.mark.skipif(not (shutil.which("rsync") and Path("/usr/bin/sqlite3").exists()
                                     and shutil.which("bash")), reason="needs bash, rsync and /usr/bin/sqlite3")

SHIMS = {
    "stat": """#!/bin/bash
args=(); fmt=""
while [ $# -gt 0 ]; do case "$1" in -f) fmt="$2"; shift 2;; *) args+=("$1"); shift;; esac; done
if [ -n "$fmt" ]; then fmt="${fmt//%m/%Y}"; fmt="${fmt//%z/%s}"; exec /usr/bin/stat "${args[@]}" -c "$fmt"; fi
exec /usr/bin/stat "${args[@]}"
""",
    "defaults": "#!/bin/bash\nexit 1\n",
    "scutil": "#!/bin/bash\necho MacBook-Pro\n",
    "pmset": "#!/bin/bash\necho \"Now drawing from 'AC Power'\"\n",
    "ssh": """#!/bin/bash
while [ $# -gt 0 ]; do case "$1" in -o|-i|-l) shift 2;; -*) shift;; *) break;; esac; done
shift
[ -n "${FAKE_PI_DOWN:-}" ] && { echo "ssh: connect: Operation timed out" >&2; exit 255; }
cd "$FAKE_PI_HOME" && exec bash -c "$*"
""",
}


@pytest.fixture
def machines(tmp_path, topics):
    shims = tmp_path / "shims"
    shims.mkdir()
    for name, body in SHIMS.items():
        (shims / name).write_text(body)
        (shims / name).chmod(0o755)
    mac, pi = tmp_path / "mac", tmp_path / "pi"
    digup = mac / "Library/Application Support/DigUp/index.noindex"
    digup.mkdir(parents=True)
    pi.mkdir()
    build_sample(digup / "index.sqlite", topics)
    (mac / ".config/hermes-digup").mkdir(parents=True)
    (mac / ".config/hermes-digup/sync.env").write_text("PI_HOST=pi@raspberrypi\n")
    env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}", "HOME": str(mac), "FAKE_PI_HOME": str(pi)}
    return digup / "index.sqlite", pi / ".local/share/hermes-digup/index", env


def run(env, *args, **extra):
    return subprocess.run(["bash", str(SCRIPT), *args], env={**env, **extra}, capture_output=True, text=True,
                          check=False,
                          timeout=60)


def add_file(db: sqlite3.Connection, path: str) -> None:
    cur = db.execute("INSERT INTO files(path, kind, size, mtime, inode, device, state) "
                     "VALUES(?, 'doc', 1, 1.8e9, 1, 1, 'done')", (path,))
    seg = db.execute("INSERT INTO segments(file, kind, modality) VALUES(?, 'name', 'none')", (cur.lastrowid,))
    db.execute("INSERT INTO fts(rowid, name, body) VALUES(?, ?, '')", (seg.lastrowid, Path(path).name))
    db.commit()


def test_sync_while_digup_writes(machines):
    source, pi_dir, env = machines
    # DigUp running: WAL mode, a committed write still only in the WAL.
    digup = sqlite3.connect(source)
    digup.execute("PRAGMA journal_mode=WAL")
    digup.execute("PRAGMA wal_autocheckpoint=0")
    add_file(digup, f"{HOME}/Documents/zanzibar-itinerary.txt")
    assert source.with_name("index.sqlite-wal").stat().st_size > 0

    result = run(env, "-v")
    assert result.returncode == 0, result.stderr
    copy = pi_dir / "index.sqlite"
    assert sqlite3.connect(copy).execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not (pi_dir / "index.sqlite-wal").exists()
    manifest = json.loads((pi_dir / "manifest.json").read_text())
    assert manifest["source_home"] == env["HOME"] and manifest["files"]["index.sqlite"]["schema"] == 5

    service = DigUpService(Settings.load({"DIGUP_INDEX_DIR": str(pi_dir), "DIGUP_CONFIG": "/none"}),
                           manager=IndexManager(pi_dir, recheck_seconds=0))
    assert service.search("zanzibar")["results"][0]["path"].endswith("zanzibar-itinerary.txt")

    # Unchanged: nothing sent, the manifest refreshed.
    again = run(env, "-v")
    assert again.returncode == 0 and "unchanged" in again.stderr

    # DigUp adds a file: the next run sends it, and the running server picks it up.
    add_file(digup, f"{HOME}/Documents/kilimanjaro-packing.txt")
    assert run(env).returncode == 0
    assert service.search("kilimanjaro")["results"][0]["path"].endswith("kilimanjaro-packing.txt")
    digup.close()


def test_pi_down_and_missing_index(machines):
    source, pi_dir, env = machines
    down = run(env, FAKE_PI_DOWN="1")
    assert down.returncode == 75
    assert not pi_dir.exists() or not (pi_dir / "index.sqlite").exists()

    source.unlink()
    missing = run(env)
    assert missing.returncode == 1 and "no DigUp index" in missing.stderr


def test_code_index_follows_digup(machines):
    source, pi_dir, env = machines
    shutil.copy(source, source.with_name("code.sqlite"))
    assert run(env).returncode == 0
    assert (pi_dir / "code.sqlite").exists()
    source.with_name("code.sqlite").unlink()   # code search turned off in DigUp
    assert run(env).returncode == 0
    assert not (pi_dir / "code.sqlite").exists()
