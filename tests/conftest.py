"""A small index with DigUp's exact schema (IndexStore.create, schema 5), and a fake embedding server."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files(
    id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, kind TEXT NOT NULL, size INTEGER NOT NULL,
    mtime REAL NOT NULL, inode INTEGER NOT NULL, device INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending', detail TEXT, info TEXT, indexed_at REAL
);
CREATE INDEX files_inode ON files(device, inode);
CREATE INDEX files_state ON files(state, kind);
CREATE TABLE segments(
    id INTEGER PRIMARY KEY, file INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, modality TEXT NOT NULL, loc REAL, loc_end REAL, excerpt TEXT, vec BLOB
);
CREATE INDEX segments_file ON segments(file);
CREATE VIRTUAL TABLE fts USING fts5(name, body, original UNINDEXED,
    tokenize = "porter unicode61 remove_diacritics 2 categories 'L* N* Co M*'");
PRAGMA user_version = 5;
"""

DIM = 768
HOME = "/Users/georges"


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


class Topics:
    """Fixed directions in vector space standing in for meanings."""

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, name: str) -> np.ndarray:
        if name not in self.cache:
            self.cache[name] = unit(self.rng.standard_normal(DIM))
        return self.cache[name]

    def near(self, name: str, noise: float = 0.3) -> np.ndarray:
        return unit(self(name) + noise * unit(self.rng.standard_normal(DIM)))


class IndexBuilder:
    def __init__(self, path: Path, topics: Topics):
        self.path = path
        self.topics = topics
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT INTO meta VALUES('vector_source', 'google/embeddinggemma-2@bfcd298 "
                        "llama.cpp-vectors1 text-q8_0+mmproj-q8_0')")

    def file(self, path: str, kind: str, state: str = "done", info: dict | None = None, mtime: float = 1.7e9) -> int:
        cur = self.db.execute(
            "INSERT INTO files(path, kind, size, mtime, inode, device, state, info) VALUES(?,?,?,?,?,?,?,?)",
            (path, kind, 1000, mtime, 1, 1, state, json.dumps(info) if info else None))
        fid = cur.lastrowid
        seg = self.db.execute("INSERT INTO segments(file, kind, modality) VALUES(?, 'name', 'none')", (fid,))
        name = Path(path).name
        folders = " ".join(Path(path).parent.relative_to(HOME).parts) if path.startswith(HOME) else ""
        self.db.execute("INSERT INTO fts(rowid, name, body) VALUES(?,?,?)", (seg.lastrowid, name, folders))
        return fid

    def segment(self, fid: int, kind: str, modality: str, vector: np.ndarray | None = None, text: str | None = None,
                loc: float | None = None, loc_end: float | None = None, excerpt: str | None = None) -> int:
        blob = vector.astype("<f2").tobytes() if vector is not None else None
        cur = self.db.execute(
            "INSERT INTO segments(file, kind, modality, loc, loc_end, excerpt, vec) VALUES(?,?,?,?,?,?,?)",
            (fid, kind, modality if vector is not None else "none", loc, loc_end,
             excerpt if excerpt is not None else (text[:240] if text else None), blob))
        if text:
            self.db.execute("INSERT INTO fts(rowid, name, body, original) VALUES(?, '', ?, NULL)",
                            (cur.lastrowid, text))
        return cur.lastrowid

    def filler(self, count: int = 120) -> None:
        """Unrelated files, so each query has a distribution to stand out from."""
        for i in range(count):
            fid = self.file(f"{HOME}/Misc/note-{i}.txt", "doc")
            self.segment(fid, "chunk", "text", self.topics.near(f"filler-{i}", 1.0), text=f"lorem ipsum {i}", loc=1)
        for i in range(40):
            fid = self.file(f"{HOME}/Pictures/IMG_{i:04}.jpg", "image")
            self.segment(fid, "image", "image", self.topics.near(f"picture-{i}", 1.0))

    def close(self) -> Path:
        self.db.commit()
        self.db.close()
        return self.path


def build_sample(path: Path, topics: Topics) -> Path:
    b = IndexBuilder(path, topics)
    b.filler()
    lease = b.file(f"{HOME}/Documents/Lease 2026.pdf", "pdf", info={"pages": "12"})
    b.segment(lease, "page", "text", topics.near("rent"), text="The monthly rent is due on the first.", loc=1)
    b.segment(lease, "page", "text", topics.near("pets"), text="Tenants may keep one cat or a small dog "
              "with written consent of the landlord.", loc=5)
    dog = b.file(f"{HOME}/Pictures/beach.heic", "image", info={"width": "4032", "height": "3024"})
    b.segment(dog, "image", "image", topics.near("dog on beach"))
    shot = b.file(f"{HOME}/Desktop/Screenshot 2026-10-01 at 9.41.07 PM.png", "screenshot")
    b.segment(shot, "image", "image", topics.near("payment error"))
    b.segment(shot, "ocr", "none", None, text="Payment declined. Error code TRX50-K7Q2LM")
    pod = b.file(f"{HOME}/Podcasts/episode 12.mp3", "audio", info={"duration": "3600"})
    b.segment(pod, "audio", "audio", topics.near("cooking", 0.8), loc=0, loc_end=30)
    b.segment(pod, "audio", "audio", topics.near("sleep"), loc=1500, loc_end=1530)
    report = b.file(f"{HOME}/Documents/Report.md", "doc")
    first = "Quarterly report. " + "Revenue grew steadily across regions. " * 40
    second = first[-150:] + " Expenses were flat and the outlook remains positive."
    b.segment(report, "chunk", "text", topics.near("revenue"), text=first, loc=1)
    b.segment(report, "chunk", "text", topics.near("expenses"), text=second, loc=2)
    return b.close()


@pytest.fixture
def topics() -> Topics:
    return Topics()


@pytest.fixture
def index_dir(tmp_path: Path, topics: Topics) -> Path:
    directory = tmp_path / "index"
    directory.mkdir()
    build_sample(directory / "index.sqlite", topics)
    (directory / "manifest.json").write_text(json.dumps({"source_home": HOME, "synced_at_epoch": 9e12}))
    return directory


class FakeEmbeddingServer:
    """Speaks the OpenAI and Ollama embedding APIs; maps prompts to vectors with `respond`.
    With `models`, only those names embed (others get oMLX's 404) and they're listed."""

    def __init__(self, respond, api_key: str | None = None, models: list[str] | None = None):
        self.respond = respond
        self.api_key = api_key
        self.models = models
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, code, payload):
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                outer.requests.append({"path": self.path, "body": None})
                names = outer.models or []
                if self.path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [{"id": n} for n in names]})
                if self.path == "/api/tags":
                    return self._json(200, {"models": [{"name": n} for n in names]})
                return self._json(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                if outer.api_key and self.headers.get("Authorization") != f"Bearer {outer.api_key}":
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b'{"error":"bad key"}')
                    return
                if outer.models is not None and body.get("model") not in outer.models:
                    return self._json(404, {"error": {"message": f"Model '{body.get('model')}' not found. "
                                                      f"Available models: {', '.join(outer.models)}",
                                                      "type": "not_found_error"}})
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vectors = [outer.respond(text).tolist() for text in inputs]
                if self.path == "/api/embed":
                    payload = {"embeddings": vectors}
                else:
                    payload = {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake_server(topics: Topics):
    def respond(prompt: str) -> np.ndarray:
        text = prompt.split("query: ", 1)[-1]
        for name in ("pets", "dog on beach", "payment error", "sleep", "revenue"):
            if name.split()[0] in text:
                return topics(name) * 3  # not normalised: the client must normalise
        return topics("unrelated query")

    server = FakeEmbeddingServer(respond)
    yield server
    server.close()
