from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from conftest import HOME, IndexBuilder, build_sample

from hermes_digup import textnorm
from hermes_digup.config import Settings
from hermes_digup.embed import Embedder, EmbeddingError
from hermes_digup.index import IndexManager, IndexUnavailable
from hermes_digup.search import merge_overlap
from hermes_digup.service import DigUpService


def service_for(index_dir: Path, urls: list[str], **extra) -> DigUpService:
    settings = Settings.load({"DIGUP_INDEX_DIR": str(index_dir), "DIGUP_EMBED_URL": ",".join(urls),
                              "DIGUP_CONFIG": "/nonexistent", **extra})
    return DigUpService(settings, manager=IndexManager(index_dir, recheck_seconds=0))


def test_meaning_finds_the_pdf_page(index_dir, fake_server):
    result = service_for(index_dir, [fake_server.url]).search("the lease clause about pets")
    top = result["results"][0]
    assert top["path"] == f"{HOME}/Documents/Lease 2026.pdf"
    assert top["where"] == {"page": 5}
    assert result["mode"] == "hybrid"
    # DigUp's query prompt, not a bare query.
    assert fake_server.requests[0]["body"]["input"] == ["task: search result | query: the lease clause about pets"]


def test_audio_hit_has_a_timestamp(index_dir, fake_server):
    top = service_for(index_dir, [fake_server.url]).search("where they talk about sleep")["results"][0]
    assert top["kind"] == "audio"
    assert top["where"]["timestamp"] == "25:00"


def test_exact_code_found_by_keywords_without_embeddings(index_dir):
    result = service_for(index_dir, []).search("TRX50-K7Q2LM")
    assert result["mode"] == "keywords only"
    top = result["results"][0]
    assert top["kind"] == "screenshot"
    assert "«" in top["excerpt"]


def test_unreachable_server_falls_back_to_keywords(index_dir):
    service = service_for(index_dir, ["http://127.0.0.1:9"], DIGUP_EMBED_TIMEOUT="1")
    result = service.search("payment declined")
    assert result["mode"] == "keywords only"
    assert any("didn't answer" in n for n in result["notes"])
    assert result["results"][0]["kind"] == "screenshot"


def test_failover_to_second_url(index_dir, fake_server):
    service = service_for(index_dir, ["http://127.0.0.1:9", fake_server.url])
    assert service.search("dog playing")["mode"] == "hybrid"


def test_kind_and_folder_filters(index_dir, fake_server):
    service = service_for(index_dir, [fake_server.url])
    assert all(r["kind"] == "image" for r in service.search("dog", kinds=["image"])["results"])
    in_pictures = service.search("the lease clause about pets", folder="~/Pictures")["results"]
    assert all(r["path"].startswith(f"{HOME}/Pictures/") for r in in_pictures)
    assert service.search("the lease clause about pets", folder="~/Documents")["results"][0]["where"] == {"page": 5}
    assert service.search("x", kinds=["spreadsheet"])["error"]


def test_read_passage_and_file(index_dir, fake_server):
    service = service_for(index_dir, [fake_server.url])
    hit = service.search("the lease clause about pets")["results"][0]
    passage = service.read_passage(hit["segment_id"], context=1)
    assert "written consent" in passage["text"]
    assert "[page 1]" in passage["text"]  # the neighbouring page
    whole = service.read_file(f"{HOME}/Documents/Report.md")
    assert whole["text"].count("Quarterly report") == 1
    assert whole["text"].endswith("outlook remains positive.")
    # The chunk overlap appears once.
    assert "Revenue grew steadily across regions. Revenue" in whole["text"]
    assert whole["text"].count("Expenses were flat") == 1
    assert service.read_file(f"{HOME}/Pictures/beach.heic")["text"] is None
    assert "error" in service.read_file("/nope")


def test_read_file_pages(index_dir):
    service = service_for(index_dir, [])
    text = service.read_file(f"{HOME}/Documents/Lease 2026.pdf", max_chars=1000)
    assert text["text"].startswith("[page 1]")
    assert text["next_offset"] is None


def test_browse(index_dir):
    service = service_for(index_dir, [])
    listed = service.browse(folder="~/Documents")
    assert {f["path"] for f in listed["files"]} == {f"{HOME}/Documents/Lease 2026.pdf", f"{HOME}/Documents/Report.md"}
    assert service.browse(kind="audio")["count"] == 1
    assert service.browse(name_contains="Screenshot 2026")["count"] == 1


def test_reload_after_replacement_and_refusing_a_broken_copy(tmp_path, index_dir, topics, fake_server):
    service = service_for(index_dir, [fake_server.url])
    before = service.search("payment declined")["results"]
    assert before

    # A newer snapshot arrives the way rsync delivers it: written aside, renamed into place.
    newer = tmp_path / "newer.sqlite"
    b = IndexBuilder(newer, topics)
    b.filler()
    fid = b.file(f"{HOME}/Downloads/invoice 4815162342.pdf", "pdf", info={"pages": "1"})
    b.segment(fid, "page", "text", topics.near("invoice"), text="Invoice 4815162342 total due", loc=1)
    b.close()
    os.replace(newer, index_dir / "index.sqlite")
    hits = service.search("4815162342")["results"]
    assert hits[0]["path"].endswith("invoice 4815162342.pdf")

    # A broken copy is refused; the loaded snapshot keeps answering.
    broken = tmp_path / "broken.sqlite"
    broken.write_bytes(b"this is not sqlite" * 100)
    os.replace(broken, index_dir / "index.sqlite")
    hits = service.search("4815162342")["results"]
    assert hits[0]["path"].endswith("invoice 4815162342.pdf")
    assert "newer_copy_refused" in service.status(check_embeddings=False)["index"]


def test_missing_index_message(tmp_path):
    service = service_for(tmp_path / "empty", [])
    with pytest.raises(IndexUnavailable, match="digup-sync"):
        service.search("anything")


def test_live_index_with_wal(tmp_path, topics, fake_server):
    """Pointing straight at DigUp's own folder (DigUp running, WAL mode) works read-only."""
    directory = tmp_path / "live"
    directory.mkdir()
    path = build_sample(directory / "index.sqlite", topics)
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("UPDATE files SET mtime = mtime")
    writer.commit()
    assert (directory / "index.sqlite-wal").exists()
    service = service_for(directory, [fake_server.url])
    assert service.search("the lease clause about pets")["results"][0]["where"] == {"page": 5}
    writer.close()


def test_fidelity_check(index_dir, topics):
    from conftest import FakeEmbeddingServer

    rows = sqlite3.connect(index_dir / "index.sqlite").execute("""
        SELECT f.path, fts.body, s.vec FROM segments s JOIN files f ON f.id = s.file JOIN fts ON fts.rowid = s.id
        WHERE s.kind = 'chunk' AND f.kind = 'doc' AND s.vec IS NOT NULL""").fetchall()
    stored = {f"title: {Path(p).stem} | text: {t}": np.frombuffer(v, dtype="<f2").astype(np.float32)
              for p, t, v in rows}
    same = FakeEmbeddingServer(lambda prompt: stored.get(prompt, topics("x")))
    other = FakeEmbeddingServer(lambda prompt: topics("x"))
    try:
        assert service_for(index_dir, [same.url]).fidelity()["verdict"] == "match"
        assert service_for(index_dir, [other.url]).fidelity()["verdict"] == "mismatch"
    finally:
        same.close()
        other.close()


def test_ollama_api_and_api_key(index_dir, topics):
    from conftest import FakeEmbeddingServer

    server = FakeEmbeddingServer(lambda prompt: topics("pets"), api_key="s3cret")
    try:
        bad = service_for(index_dir, [server.url], DIGUP_EMBED_API="ollama")
        with pytest.raises(EmbeddingError, match="API_KEY"):
            bad.embedder.embed_query("x")
        good = service_for(index_dir, [server.url], DIGUP_EMBED_API="ollama", DIGUP_EMBED_API_KEY="s3cret")
        assert good.search("pets")["results"][0]["where"] == {"page": 5}
        assert server.requests[-1]["path"] == "/api/embed"
    finally:
        server.close()


def test_wrong_dimension_is_explained(index_dir):
    from conftest import FakeEmbeddingServer

    server = FakeEmbeddingServer(lambda prompt: np.ones(1024, dtype=np.float32))
    try:
        embedder = Embedder(Settings.load({"DIGUP_EMBED_URL": server.url, "DIGUP_CONFIG": "/nonexistent"}))
        with pytest.raises(EmbeddingError, match="1024 dimensions"):
            embedder.embed_query("x")
    finally:
        server.close()


def test_query_cache(index_dir, fake_server):
    service = service_for(index_dir, [fake_server.url])
    service.search("dog")
    service.search("dog")
    assert len(fake_server.requests) == 1


def test_config_file(tmp_path):
    config = tmp_path / "config.env"
    config.write_text("# comment\nDIGUP_EMBED_URL='http://a:1, http://b:2/'\nexport DIGUP_STALE_HOURS=5\n")
    settings = Settings.load({"DIGUP_CONFIG": str(config), "DIGUP_STALE_HOURS": "7"})
    assert settings.embed_urls == ["http://a:1", "http://b:2"]
    assert settings.stale_hours == 7  # the environment wins


def test_textnorm():
    assert textnorm.terms("Find the PHOTO of my dog on the beach") == ["dog", "beach"]
    assert textnorm.is_lookup(["trx50"]) and not textnorm.is_lookup(["dog"])
    assert textnorm.kind_hints("zebra in a video") == {"video"}
    assert textnorm.normalized("ﬁle‍") == "file"
    assert textnorm.normalized("أحمد") == "احمد"
    assert textnorm.snippet("Payment declined. Error code X", ["declined"]) == "Payment «declined». Error code X"


def test_merge_overlap():
    a = "one two three four five six seven eight nine ten eleven twelve"
    assert merge_overlap(a, "seven eight nine ten eleven twelve thirteen") == "thirteen"
    assert merge_overlap(a, "something else entirely") == "something else entirely"


def test_hanging_server_is_not_retried(index_dir):
    """A server that accepts but never answers costs one timeout, then search uses keywords."""
    import socket
    import time

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        service = service_for(index_dir, [f"http://127.0.0.1:{listener.getsockname()[1]}"],
                              DIGUP_EMBED_TIMEOUT="1")
        started = time.monotonic()
        result = service.search("payment declined")
        assert time.monotonic() - started < 1.9
        assert result["mode"] == "keywords only"
        assert result["results"][0]["kind"] == "screenshot"
    finally:
        listener.close()
