"""Reading DigUp's index: one SQLite file per index (index.sqlite, and code.sqlite for code search).

The layout is DigUp's (Sources/DigUpKit/IndexStore.swift, schema 5): `files`, `segments` with a
768 x float16 L2-normalised vector per segment, an FTS5 table `fts` keyed by segment id, and `meta`.
Nothing here ever writes to it.

A snapshot is loaded whole (vectors into memory, as DigUp does) and swapped for a newer one only
after the newer one has opened and passed validation, so a bad or half-copied file never takes
search down: the previous snapshot keeps answering and the error shows in `index_status`.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DIMENSION = 768
VECTOR_BYTES = DIMENSION * 2
MAIN_FILE = "index.sqlite"
CODE_FILE = "code.sqlite"
MANIFEST_FILE = "manifest.json"
# Schemas this reader understands. A newer DigUp may add columns; it's tried, with a warning.
KNOWN_SCHEMAS = range(4, 6)
REQUIRED_TABLES = {"files", "segments", "fts", "meta"}

MODALITIES = ("none", "text", "image", "audio")
# Long files' later steps are scored but don't set the per-query scale (Searcher.reload).
BIG_PDF_PAGES, PDF_STEP, DOC_STEP = 50, 30, 40
LIVE_RECHECK_SECONDS = 60.0


class IndexUnavailable(RuntimeError):
    pass


def _stat_key(path: Path) -> tuple | None:
    """What changes when the file is replaced (a new inode) or written in place (a live index)."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    key: tuple = (st.st_ino, st.st_size, st.st_mtime_ns)
    wal = path.with_name(path.name + "-wal")
    try:
        ws = wal.stat()
        key += (ws.st_size, ws.st_mtime_ns)
    except FileNotFoundError:
        pass
    return key


def _float32_fits(nbytes: int) -> bool:
    """Keep vectors as float32 when that takes under a quarter of the memory that's free
    (DIGUP_FLOAT32=0/1 decides instead). A 100k-vector index is 290 MB as float32, 145 MB as float16."""
    forced = os.environ.get("DIGUP_FLOAT32")
    if forced in ("0", "1"):
        return forced == "1"
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return nbytes < int(line.split()[1]) * 1024 // 4
    except (OSError, ValueError, IndexError):
        pass
    return nbytes < 256 * 1024 * 1024  # no /proc (macOS): only small indexes


def connect(path: Path) -> sqlite3.Connection:
    """Read-only. A synced snapshot is opened immutable (no locks, no -wal/-shm); a live DigUp index
    (DigUp running on this machine, with its write-ahead log) is opened read-only alongside it."""
    live = path.with_name(path.name + "-wal").exists()
    uri = path.resolve().as_uri() + ("?mode=ro" if live else "?mode=ro&immutable=1")
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA query_only = 1")
    return conn


@dataclass
class Snapshot:
    path: Path
    conn: sqlite3.Connection
    stat_key: tuple | None
    schema: int
    meta: dict[str, str]
    vectors: np.ndarray            # rows x 768, float32 or float16 (as stored)
    row_file: np.ndarray           # file id per row
    row_file_index: np.ndarray     # dense 0..n_files-1 per row
    row_segment: np.ndarray        # segment id per row
    row_modality: np.ndarray       # index into MODALITIES
    row_sets_scale: np.ndarray     # bool
    file_ids: np.ndarray           # dense index -> file id
    file_kind: dict[int, str]
    loaded_at: float = field(default_factory=time.time)
    warnings: list[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def live(self) -> bool:
        return self.stat_key is not None and len(self.stat_key) > 3   # it has a write-ahead log beside it

    @property
    def is_code(self) -> bool:
        return self.path.name == CODE_FILE

    def query(self, sql: str, params: tuple | list = ()) -> list[tuple]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    @classmethod
    def load(cls, path: Path) -> Snapshot:
        key = _stat_key(path)
        if key is None:
            raise IndexUnavailable(f"no index at {path}")
        conn = connect(path)
        try:
            return cls._load(path, conn, key)
        except Exception:
            conn.close()
            raise

    @classmethod
    def _load(cls, path: Path, conn: sqlite3.Connection, key: tuple) -> Snapshot:
        warnings: list[str] = []
        try:
            schema = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        except sqlite3.DatabaseError as error:
            raise IndexUnavailable(f"{path} isn't a readable SQLite file ({error})") from error
        missing = REQUIRED_TABLES - tables
        if missing:
            raise IndexUnavailable(f"{path} isn't a DigUp index (no {', '.join(sorted(missing))} table)")
        if schema not in KNOWN_SCHEMAS:
            warnings.append(f"index schema {schema} is newer than this server knows ({KNOWN_SCHEMAS.stop - 1}); "
                            "results may be off until hermes-digup is updated")
        meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())

        file_kind = {fid: kind for fid, kind in conn.execute("SELECT id, kind FROM files")}
        rows = conn.execute("""
            SELECT s.id, s.file, s.modality, s.vec, s.kind, s.loc, f.kind,
                   CAST(json_extract(f.info, '$.pages') AS INTEGER)
            FROM segments s JOIN files f ON f.id = s.file WHERE s.vec IS NOT NULL ORDER BY s.id
            """)
        seg, fil, mod, scale, blobs = [], [], [], [], []
        skipped = 0
        modality_index = {m: i for i, m in enumerate(MODALITIES)}
        for sid, fid, modality, vec, skind, loc, fkind, pages in rows:
            if vec is None or len(vec) != VECTOR_BYTES:
                skipped += 1
                continue
            place = int(loc or 0)
            if fkind == "pdf":
                later = skind != "chunk" and (pages or 0) > BIG_PDF_PAGES and place > PDF_STEP
            else:
                later = skind == "chunk" and place > DOC_STEP
            seg.append(sid)
            fil.append(fid)
            mod.append(modality_index.get(modality, 0))
            scale.append(not later)
            blobs.append(vec)
        if skipped:
            warnings.append(f"{skipped} vectors had an unexpected size and were left out")
        vectors = (np.frombuffer(b"".join(blobs), dtype="<f2").reshape(-1, DIMENSION)
                   if blobs else np.zeros((0, DIMENSION), dtype=np.float16))
        del blobs
        if _float32_fits(vectors.size * 4):
            # Scoring float32 directly is ~3x faster than converting float16 on every query.
            vectors = vectors.astype(np.float32)
        row_file = np.asarray(fil, dtype=np.int64)
        file_ids, row_file_index = np.unique(row_file, return_inverse=True)
        snapshot = cls(
            path=path, conn=conn, stat_key=key, schema=schema, meta=meta, vectors=vectors,
            row_file=row_file, row_file_index=row_file_index.astype(np.int64),
            row_segment=np.asarray(seg, dtype=np.int64), row_modality=np.asarray(mod, dtype=np.int8),
            row_sets_scale=np.asarray(scale, dtype=bool), file_ids=file_ids, file_kind=file_kind,
            warnings=warnings,
        )
        # One FTS query proves the full-text index opens with this machine's SQLite (FTS5, the tokenizer).
        try:
            snapshot.query("SELECT rowid FROM fts WHERE fts MATCH ? LIMIT 1", ('"test"',))
        except sqlite3.Error as error:
            raise IndexUnavailable(f"keyword index in {path} can't be read by this SQLite "
                                   f"({sqlite3.sqlite_version}): {error}") from error
        return snapshot

    def counts(self) -> dict:
        by_kind: dict[str, dict[str, int]] = {}
        for kind, state, n in self.query("SELECT kind, state, COUNT(*) FROM files GROUP BY kind, state"):
            by_kind.setdefault(kind, {})[state] = n
        return {"files_by_kind": by_kind, "vectors": len(self.vectors)}


class IndexManager:
    """The current snapshot of each index file, reloaded when the file on disk changes."""

    def __init__(self, index_dir: Path, recheck_seconds: float = 3.0):
        self.index_dir = index_dir
        self.recheck_seconds = recheck_seconds
        self._snapshots: dict[str, Snapshot] = {}
        self._checked: dict[str, float] = {}
        self._errors: dict[str, str] = {}
        self._refused: dict[str, tuple | None] = {}
        self._lock = threading.Lock()

    def path(self, code: bool = False) -> Path:
        return self.index_dir / (CODE_FILE if code else MAIN_FILE)

    def get(self, code: bool = False) -> Snapshot:
        name = CODE_FILE if code else MAIN_FILE
        with self._lock:
            now = time.monotonic()
            current = self._snapshots.get(name)
            # A live index (DigUp running here) changes with every file it reads: reload it at most once a minute.
            every = max(self.recheck_seconds, LIVE_RECHECK_SECONDS) if current is not None and current.live \
                else self.recheck_seconds
            if current is not None and now - self._checked.get(name, 0) < every:
                return current
            self._checked[name] = now
            path = self.path(code)
            key = _stat_key(path)
            if current is not None and (key == current.stat_key or key is None):
                # Unchanged, or the file vanished mid-sync: keep answering from what's loaded.
                return current
            if current is not None and name in self._refused and key == self._refused[name]:
                return current  # the same file that was refused before: don't load it again
            try:
                fresh = Snapshot.load(path)
            except Exception as error:  # keep the old snapshot; report why the new one was refused
                message = str(error)
                if self._errors.get(name) != message:
                    log.warning("not loading %s: %s", path, message)
                self._errors[name] = message
                self._refused[name] = key
                if current is not None:
                    return current
                if key is None:
                    if code:
                        raise IndexUnavailable(
                            f"No DigUp code index at {path}: code search is off in DigUp (Settings > Code), "
                            "or hasn't been synced yet.") from None
                    raise IndexUnavailable(
                        f"No DigUp index at {path} yet. Run the sync on the Mac that has DigUp "
                        "(deploy/mac-laptop/digup-sync.sh), or set DIGUP_INDEX_DIR.") from None
                raise IndexUnavailable(message) from error
            self._errors.pop(name, None)
            self._refused.pop(name, None)
            self._snapshots[name] = fresh
            log.info("loaded %s: %d vectors, %d files", path, len(fresh.vectors), len(fresh.file_kind))
            if current is not None:
                # Searches already running on the old one finish first: closing waits for its lock.
                threading.Thread(target=_close_later, args=(current,), daemon=True).start()
            return fresh

    def error(self, code: bool = False) -> str | None:
        return self._errors.get(CODE_FILE if code else MAIN_FILE)

    def manifest(self) -> dict:
        try:
            return json.loads((self.index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def age_hours(self, code: bool = False) -> float | None:
        """Hours since the index was last synced (the manifest's time, else the file's)."""
        synced = self.manifest().get("synced_at_epoch")
        if not isinstance(synced, (int, float)):
            try:
                synced = os.path.getmtime(self.path(code))
            except OSError:
                return None
        return max(0.0, (time.time() - synced) / 3600)


def _close_later(snapshot: Snapshot) -> None:
    time.sleep(30)
    with snapshot.lock:
        snapshot.close()
