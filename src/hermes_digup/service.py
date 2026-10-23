"""What the tools do, independent of MCP, so the CLI and the tests use the same code."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import search as searching
from .config import Settings
from .embed import Embedder, EmbeddingError, document_prompt
from .index import IndexManager, IndexUnavailable, Snapshot

log = logging.getLogger(__name__)

MAX_LIMIT = 50
MAX_CHARS = 60_000


class DigUpService:
    def __init__(self, settings: Settings, manager: IndexManager | None = None, embedder: Embedder | None = None):
        self.settings = settings
        self.manager = manager or IndexManager(settings.index_dir)
        self.embedder = embedder or Embedder(settings)

    # MARK: search

    def search(self, query: str, kinds: list[str] | None = None, limit: int = 10, folder: str | None = None,
               code: bool = False) -> dict:
        query = (query or "").strip()
        if code and query.lower().startswith("code:"):
            query = query[5:].strip()
        elif not code and query.lower().startswith("code:"):
            code, query = True, query[5:].strip()
        if not query:
            return {"error": "empty query"}
        limit = max(1, min(int(limit or 10), MAX_LIMIT))
        bad = [k for k in kinds or [] if k not in searching.FILE_KINDS]
        if bad:
            return {"error": f"unknown kind(s) {bad}; use any of {list(searching.FILE_KINDS)}"}
        snapshot = self.manager.get(code=code)
        notes: list[str] = list(snapshot.warnings)

        vector: np.ndarray | None = None
        mode = "hybrid"
        started = time.monotonic()
        if not self.embedder.configured:
            mode = "keywords only"
            notes.append("No embedding server configured (DIGUP_EMBED_URL), so only exact words were matched.")
        else:
            try:
                vector = self.embedder.embed_query(query, code=code)
            except EmbeddingError as error:
                mode = "keywords only"
                notes.append("The embedding server didn't answer, so only exact words were matched "
                             f"(meaning search resumes when it's back): {error}")
        embed_ms = round((time.monotonic() - started) * 1000)

        hits = searching.search(snapshot, query, vector, kinds=set(kinds) if kinds else None, limit=limit,
                                folder=self._mac_path(folder))
        age = self.manager.age_hours(code)
        if age is not None and age > self.settings.stale_hours:
            notes.append(f"The index was last synced {age:.0f} hours ago; files added since then aren't in it.")
        result = {
            "query": query,
            "index": "code" if code else "files",
            "mode": mode,
            "results": [hit.to_dict() for hit in hits],
            "embed_ms": embed_ms if vector is not None else None,
        }
        if not hits:
            notes.append("Nothing stood out for this query. Try describing it differently, fewer words, "
                         "or exact words that are in the file.")
        if notes:
            result["notes"] = notes
        return result

    def _mac_path(self, folder: str | None) -> str | None:
        """Paths in the index are the Mac's; "~/Downloads" means the Mac user's Downloads."""
        if not folder:
            return None
        folder = folder.strip()
        home = self.manager.manifest().get("source_home")
        if folder.startswith("~") and home:
            folder = home.rstrip("/") + folder[1:]
        return folder

    # MARK: reading

    def _find_file(self, snapshot: Snapshot, path: str) -> tuple[int, str, str] | None:
        rows = snapshot.query("SELECT id, path, kind FROM files WHERE path = ?", (path,))
        if not rows:
            mac = self._mac_path(path)
            rows = snapshot.query("SELECT id, path, kind FROM files WHERE path = ?", (mac,)) if mac else []
        return rows[0] if rows else None

    def read_passage(self, segment_id: int, context: int = 0, code: bool = False) -> dict:
        snapshot = self.manager.get(code=code)
        rows = snapshot.query("""
            SELECT s.file, s.kind, s.loc, s.loc_end, f.path, f.kind FROM segments s JOIN files f ON f.id = s.file
            WHERE s.id = ?""", (int(segment_id),))
        if not rows:
            return {"error": f"no segment {segment_id} in the index (it may have changed since the search: "
                             "search again)"}
        file_id, kind, loc, loc_end, path, file_kind = rows[0]
        context = max(0, min(int(context or 0), 3))
        pieces = []
        if context and kind in searching.TEXT_SEGMENTS:
            all_pieces = searching.file_text(snapshot, file_id)
            here = next((i for i, p in enumerate(all_pieces) if p[0] == kind and p[1] == loc), None)
            if here is not None:
                pieces = all_pieces[max(0, here - context):here + context + 1]
        if not pieces:
            text = searching.segment_text(snapshot, int(segment_id))
            pieces = [(kind, loc, text)] if text else []
        if not pieces:
            what = {"image": "a picture", "frame": "a video frame", "audio": "a stretch of audio"}.get(kind, kind)
            return {"path": path, "kind": file_kind, "where": searching.describe_place(file_kind, kind, loc, loc_end),
                    "text": None, "note": f"This match is {what}: DigUp keeps its meaning, not text, for it."}
        text = "\n\n".join(_label(k, l) + t for k, l, t in pieces)
        return {"path": path, "kind": file_kind, "where": searching.describe_place(file_kind, kind, loc, loc_end),
                "text": text[:MAX_CHARS], "truncated": len(text) > MAX_CHARS}

    def read_file(self, path: str, offset: int = 0, max_chars: int = 20_000, code: bool = False) -> dict:
        snapshot = self.manager.get(code=code)
        found = self._find_file(snapshot, path)
        if not found:
            return {"error": f"{path} isn't in the index. Use the exact path a search returned."}
        file_id, real_path, kind = found
        pieces = searching.file_text(snapshot, file_id)
        if not pieces:
            note = {"image": "it's a picture", "audio": "it's audio", "video": "it's a video",
                    "screenshot": "no text was read from this screenshot"}.get(kind, "no text was stored for it")
            return {"path": real_path, "kind": kind, "text": None,
                    "note": f"The index has no text for this file: {note}. Its matches carry meaning, not words."}
        whole = "\n\n".join(_label(k, l) + t for k, l, t in pieces)
        offset = max(0, int(offset or 0))
        max_chars = max(1000, min(int(max_chars or 20_000), MAX_CHARS))
        part = whole[offset:offset + max_chars]
        end = offset + len(part)
        state = snapshot.query("SELECT state, detail FROM files WHERE id = ?", (file_id,))[0]
        result = {"path": real_path, "kind": kind, "text": part, "offset": offset, "total_chars": len(whole),
                  "next_offset": end if end < len(whole) else None}
        if state[0] == "partial":
            result["note"] = f"DigUp hasn't finished reading this file ({state[1]}); later parts may be missing."
        return result

    def browse(self, folder: str | None = None, kind: str | None = None, name_contains: str | None = None,
               limit: int = 20, code: bool = False) -> dict:
        snapshot = self.manager.get(code=code)
        sql, params = "SELECT path, kind, mtime, size, state FROM files WHERE 1", []
        folder = self._mac_path(folder)
        if folder:
            prefix = folder.rstrip("/") + "/"
            sql += " AND path >= ? AND path < ?"
            params += [prefix, prefix + "\U0010ffff"]
        if kind:
            if kind not in searching.FILE_KINDS:
                return {"error": f"unknown kind {kind!r}; use one of {list(searching.FILE_KINDS)}"}
            sql += " AND kind = ?"
            params.append(kind)
        if name_contains:
            sql += " AND path LIKE ? ESCAPE '\\'"
            escaped = name_contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        limit = max(1, min(int(limit or 20), 200))
        sql += " ORDER BY mtime DESC LIMIT ?"
        params.append(limit)
        files = [{"path": p, "kind": k, "modified": searching.iso(m), "size_bytes": s, "state": st}
                 for p, k, m, s, st in snapshot.query(sql, params)]
        return {"files": files, "count": len(files)}

    # MARK: status

    def status(self, check_embeddings: bool = True) -> dict:
        out: dict = {"index_dir": str(self.manager.index_dir)}
        manifest = self.manager.manifest()
        if manifest:
            out["last_sync"] = {k: manifest[k] for k in ("synced_at", "source_host", "digup_index_dir")
                                if k in manifest}
        for code in (False, True):
            name = "code_index" if code else "index"
            if not self.manager.path(code).exists() and code:
                continue
            try:
                snapshot = self.manager.get(code=code)
                entry = {"file": str(snapshot.path), "schema": snapshot.schema, **snapshot.counts(),
                         "vectors_made_by": snapshot.meta.get("vector_source"),
                         "loaded_at": searching.iso(snapshot.loaded_at)}
                age = self.manager.age_hours(code)
                if age is not None:
                    entry["age_hours"] = round(age, 1)
                    if age > self.settings.stale_hours:
                        entry["stale"] = True
                if snapshot.warnings:
                    entry["warnings"] = snapshot.warnings
                if self.manager.error(code):
                    entry["newer_copy_refused"] = self.manager.error(code)
            except IndexUnavailable as error:
                entry = {"error": str(error)}
            out[name] = entry
        out["embedding"] = {"api": self.settings.embed_api, "model": self.settings.embed_model,
                            "servers": self.embedder.health() if check_embeddings and self.embedder.configured
                            else self.settings.embed_urls}
        if not self.embedder.configured:
            out["embedding"]["note"] = "DIGUP_EMBED_URL isn't set: searches match exact words only."
        return out

    def fidelity(self, samples: int = 6) -> dict:
        """Embed a few stored document chunks again, exactly as DigUp did, and compare with the stored
        vectors: ~1.0 means the embedding server runs DigUp's model and settings."""
        snapshot = self.manager.get()
        rows = snapshot.query("""
            SELECT s.id, f.path, coalesce(fts.original, fts.body) FROM segments s
            JOIN files f ON f.id = s.file JOIN fts ON fts.rowid = s.id
            WHERE s.kind = 'chunk' AND f.kind = 'doc' AND s.vec IS NOT NULL
            ORDER BY s.id DESC LIMIT ?""", (int(samples),))
        if not rows:
            return {"checked": 0, "note": "the index has no document chunks to compare"}
        prompts = [document_prompt(Path(path).stem, text) for _, path, text in rows]
        vectors = self.embedder.embed_many(prompts)
        index_of = {int(s): i for i, s in enumerate(snapshot.row_segment)}
        cosines = []
        for (sid, _, _), vector in zip(rows, vectors):
            row = index_of.get(int(sid))
            if row is not None:
                cosines.append(float(snapshot.vectors[row].astype(np.float32) @ vector))
        mean = sum(cosines) / len(cosines) if cosines else 0.0
        verdict = "match" if mean >= 0.98 else "close" if mean >= 0.9 else "mismatch"
        return {"checked": len(cosines), "mean_cosine": round(mean, 4), "min_cosine": round(min(cosines), 4)
                if cosines else None, "verdict": verdict}


def _label(kind: str, loc: float | None) -> str:
    if loc is None:
        return ""
    if kind in ("page", "page_image"):
        return f"[page {int(loc)}]\n"
    if kind == "lines":
        return f"[line {int(loc)}]\n"
    return ""
