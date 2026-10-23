"""Hybrid search over a DigUp snapshot, ported from DigUp's Searcher.swift.

Meaning: cosine similarity of the query vector with every stored vector, centred per modality
(text, image and audio vectors sit at different levels) and put in standard deviations, so only
segments that stand out (z >= 2) count. Words: FTS5 over file names, folders, page and document
text and screenshot OCR. A file qualifies by standing out in meaning or by holding at least half
the query's words; meaning z plus a word-coverage bonus orders them. Ranking matches DigUp's.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import textnorm
from .index import MODALITIES, Snapshot

WORD_WEIGHT = 2.0
CODE_WORD_WEIGHT = 1.0
Z_GATE = 2.0
STRONG_RATIO = 0.5
BLOCK_ROWS = 8192
TEXT_SEGMENTS = ("page", "page_image", "chunk", "ocr", "lines", "cells")
FILE_KINDS = ("screenshot", "image", "pdf", "doc", "audio", "video", "code")


@dataclass
class Hit:
    file_id: int
    path: str
    kind: str
    score: float
    z: float | None
    cosine: float | None
    keyword_rank: int | None
    segment_id: int
    segment: str
    loc: float | None
    loc_end: float | None
    excerpt: str | None
    modified: float
    size: int
    info: dict
    weak: bool = False

    def to_dict(self) -> dict:
        matched = [m for m, on in (("meaning", self.z is not None and self.z >= Z_GATE),
                                   ("words", self.keyword_rank is not None)) if on]
        out = {
            "path": self.path,
            "name": Path(self.path).name,
            "kind": self.kind,
            "score": round(self.score, 3),
            "matched_by": matched or ["meaning"],
            "where": describe_place(self.kind, self.segment, self.loc, self.loc_end),
            "excerpt": self.excerpt,
            "modified": iso(self.modified),
            "size_bytes": self.size,
            "segment_id": self.segment_id,
        }
        details = {k: self.info[k] for k in ("pages", "duration", "taken", "width", "height") if k in self.info}
        if details:
            out["details"] = details
        if self.weak:
            out["weaker_match"] = True
        return out


def iso(epoch: float | None) -> str | None:
    if not epoch:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="seconds")


def timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def describe_place(kind: str, segment: str, loc: float | None, loc_end: float | None) -> dict | None:
    """Where in the file the match is: a page, a moment, a chunk or lines."""
    if loc is None:
        return None
    if segment in ("page", "page_image"):
        return {"page": int(loc)}
    if segment in ("audio", "frame"):
        place = {"seconds": round(loc, 1), "timestamp": timestamp(loc)}
        if loc_end is not None:
            place["until"] = timestamp(loc_end)
        return place
    if segment == "chunk":
        return {"chunk": int(loc)}
    if segment == "lines":
        return {"lines": [int(loc), int(loc_end if loc_end is not None else loc)]}
    if segment == "cells":
        return {"cells": [int(loc), int(loc_end if loc_end is not None else loc)]}
    return {"loc": loc}


def scores(snapshot: Snapshot, query: np.ndarray) -> np.ndarray:
    """Cosine with every stored vector (all L2-normalised); float16 is scored a block at a time."""
    rows = len(snapshot.vectors)
    q = query.astype(np.float32)
    if snapshot.vectors.dtype == np.float32:
        return snapshot.vectors @ q
    out = np.empty(rows, dtype=np.float32)
    for start in range(0, rows, BLOCK_ROWS):
        block = snapshot.vectors[start:start + BLOCK_ROWS].astype(np.float32)
        out[start:start + len(block)] = block @ q
    return out


def meaning(cos: np.ndarray, modality: np.ndarray, sets_scale: np.ndarray) -> np.ndarray:
    """Each score minus its modality's mean, in standard deviations of all of them (Searcher.meaning).
    Modalities with fewer than 20 scale-setting vectors take the overall mean."""
    cos = cos.astype(np.float64)
    scale = cos[sets_scale] if sets_scale.any() else cos
    scale_mod = modality[sets_scale] if sets_scale.any() else modality
    overall = float(scale.mean()) if len(scale) else 0.0
    means = np.full(len(MODALITIES), overall)
    for m in range(len(MODALITIES)):
        group = scale[scale_mod == m]
        if len(group) >= 20:
            means[m] = group.mean()
    centred = cos - means[modality]
    ref = centred[sets_scale] if sets_scale.any() else centred
    mean = float(ref.mean()) if len(ref) else 0.0
    deviation = max(float(ref.std()) if len(ref) else 0.0, 0.02)
    return (centred - mean) / deviation


def _is_indexed(snapshot: Snapshot, pattern: str) -> bool:
    return bool(snapshot.query("SELECT 1 FROM fts WHERE fts MATCH ? LIMIT 1", (pattern,)))


def search(snapshot: Snapshot, text: str, query_vector: np.ndarray | None, kinds: set[str] | None = None,
           limit: int = 10, folder: str | None = None) -> list[Hit]:
    word_weight = CODE_WORD_WEIGHT if snapshot.is_code else WORD_WEIGHT
    folder_prefix = (folder.rstrip("/") + "/") if folder else None
    allowed_files: set[int] | None = None
    if kinds or folder_prefix:
        allowed_files = {fid for fid, kind in snapshot.file_kind.items() if not kinds or kind in kinds}
        if folder_prefix:
            inside = {r[0] for r in snapshot.query(
                "SELECT id FROM files WHERE path >= ? AND path < ?", (folder_prefix, folder_prefix + "\U0010ffff"))}
            allowed_files &= inside

    # Meaning: each file's best segment on this query's scale.
    best: dict[int, tuple[float, float, int]] = {}
    if query_vector is not None and len(snapshot.vectors):
        cos = scores(snapshot, query_vector)
        z = meaning(cos, snapshot.row_modality, snapshot.row_sets_scale)
        mask = np.ones(len(z), dtype=bool)
        if allowed_files is not None:
            mask = np.isin(snapshot.row_file, np.fromiter(allowed_files, dtype=np.int64, count=len(allowed_files)))
        rows = np.nonzero(mask)[0]
        if len(rows):
            # Best row per file: sort by (file, -z) and take each file's first.
            order = rows[np.lexsort((-z[rows], snapshot.row_file_index[rows]))]
            files = snapshot.row_file_index[order]
            first = order[np.concatenate(([True], files[1:] != files[:-1]))]
            for row in first:
                best[int(snapshot.row_file[row])] = (float(z[row]), float(cos[row]), int(row))

    # Words.
    words = textnorm.terms(text)
    pats = textnorm.patterns(words)
    for i, word in enumerate(words):
        if len(word) >= 3 and not pats[i].endswith("*") and not _is_indexed(snapshot, pats[i]):
            pats[i] += "*"  # in no file as typed: match it as the start of longer words
    keyword: list[tuple[int, int, str | None]] = []
    coverage: dict[int, float] = {}
    if pats:
        seen: set[int] = set()
        try:
            rows = snapshot.query("""
                SELECT s.file, s.id, snippet(fts, 1, '«', '»', '…', 12), fts.original FROM fts
                JOIN segments s ON s.id = fts.rowid
                WHERE fts MATCH ? ORDER BY bm25(fts, 3.0, 1.0) LIMIT 1000
                """, (" OR ".join(pats),))
        except sqlite3.OperationalError:
            rows = []
        for fid, sid, snip, original in rows:
            if len(keyword) >= 100:
                break
            if (allowed_files is not None and fid not in allowed_files) or fid in seen:
                continue
            seen.add(fid)
            keyword.append((fid, sid, (textnorm.snippet(original, words) if original else None) or snip))
        if keyword:
            ids = ",".join(str(k[0]) for k in keyword)
            for pattern in pats:
                for (fid,) in snapshot.query(f"""
                        SELECT DISTINCT s.file FROM fts JOIN segments s ON s.id = fts.rowid
                        WHERE fts MATCH ? AND s.file IN ({ids})""", (pattern,)):
                    coverage[fid] = coverage.get(fid, 0.0) + 1 / len(pats)

    lookup = textnorm.is_lookup(words)
    meaning_weight, words_weight = (0.5, word_weight * 2) if lookup else (1.0, word_weight)
    hinted = textnorm.kind_hints(text)
    candidates = {f for f, (zv, _, _) in best.items() if zv >= Z_GATE}
    candidates |= {f for f, share in coverage.items() if share >= 0.5}
    scored = {}
    for fid in candidates:
        zv = best[fid][0] if fid in best else 0.0
        score = meaning_weight * max(zv, 0.0) + words_weight * coverage.get(fid, 0.0) ** 1.5
        if snapshot.file_kind.get(fid) in hinted:
            score += 1
        scored[fid] = score

    word_rank = {}
    for rank, (fid, sid, snip) in enumerate(keyword, start=1):
        word_rank.setdefault(fid, (rank, sid, snip))

    hits: list[Hit] = []
    for fid, score in sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
        semantic = best.get(fid)
        kw = word_rank.get(fid)
        use_meaning = semantic is not None and (semantic[0] >= Z_GATE or kw is None)
        segment_id = int(snapshot.row_segment[semantic[2]]) if use_meaning else (kw[1] if kw else None)
        if segment_id is None:
            continue
        frow = snapshot.query("SELECT path, kind, mtime, size, info FROM files WHERE id = ?", (fid,))
        srow = snapshot.query("SELECT kind, loc, loc_end, excerpt FROM segments WHERE id = ?", (segment_id,))
        if not frow or not srow:
            continue
        path, kind, mtime, size, info_json = frow[0]
        seg_kind, loc, loc_end, stored_excerpt = srow[0]
        # Quote matched words only from the segment we point at (a hit on page 5 never shows page 2's words).
        snip = kw[2] if kw and kw[1] == segment_id and kw[2] and "«" in kw[2] else None
        if snip is None and stored_excerpt is None and loc is None and kw and kw[2] and "«" in kw[2]:
            matched_kind = snapshot.query("SELECT kind FROM segments WHERE id = ?", (kw[1],))
            if matched_kind and matched_kind[0][0] != "name":
                snip = kw[2]
        try:
            info = json.loads(info_json) if info_json else {}
        except ValueError:
            info = {}
        hits.append(Hit(
            file_id=fid, path=path, kind=kind, score=score,
            z=semantic[0] if semantic else None, cosine=semantic[1] if semantic else None,
            keyword_rank=kw[0] if kw else None, segment_id=segment_id, segment=seg_kind, loc=loc,
            loc_end=loc_end, excerpt=textnorm.one_line(snip or stored_excerpt or "") or None,
            modified=mtime, size=size, info=info if isinstance(info, dict) else {},
        ))
    if hits and hits[0].score > 0:
        for hit in hits:
            hit.weak = hit.score < hits[0].score * STRONG_RATIO
    return hits


def segment_text(snapshot: Snapshot, segment_id: int) -> str | None:
    rows = snapshot.query("SELECT coalesce(original, body) FROM fts WHERE rowid = ?", (segment_id,))
    return rows[0][0] if rows else None


def merge_overlap(previous: str, current: str, max_overlap: int = 400) -> str:
    """DigUp's document chunks overlap by up to ~200 characters; drop the repeat."""
    limit = min(max_overlap, len(previous), len(current))
    for size in range(limit, 20, -1):
        if previous.endswith(current[:size]):
            return current[size:].lstrip()
    return current


def file_text(snapshot: Snapshot, file_id: int) -> list[tuple[str, float | None, str]]:
    """Every stored text piece of a file, in reading order: (segment kind, loc, text)."""
    marks = ",".join("?" * len(TEXT_SEGMENTS))
    rows = snapshot.query(f"""
        SELECT s.kind, s.loc, coalesce(fts.original, fts.body) FROM segments s JOIN fts ON fts.rowid = s.id
        WHERE s.file = ? AND s.kind IN ({marks}) ORDER BY coalesce(s.loc, 0), s.id
        """, (file_id, *TEXT_SEGMENTS))
    pieces: list[tuple[str, float | None, str]] = []
    seen_pages: set[float] = set()
    for kind, loc, text in rows:
        if not text:
            continue
        if kind in ("page", "page_image"):
            # A page can have a text segment and an image segment carrying the same words.
            if loc in seen_pages:
                continue
            seen_pages.add(loc)
        if kind == "chunk" and pieces and pieces[-1][0] == "chunk":
            text = merge_overlap(pieces[-1][2], text)
        pieces.append((kind, loc, text))
    return pieces
