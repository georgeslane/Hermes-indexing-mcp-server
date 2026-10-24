"""hermes-digup: run the MCP server, or search and check the setup from a shell."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__
from .config import Settings
from .embed import EmbeddingError
from .index import IndexUnavailable
from .service import DigUpService


def _print(data: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    if "error" in data:
        print(f"error: {data['error']}")
        return
    for note in data.get("notes", []):
        print(f"note: {note}")
    for i, hit in enumerate(data.get("results", []), start=1):
        where = hit.get("where") or {}
        place = ", ".join(f"{k} {v}" for k, v in where.items() if k != "seconds")
        weak = "  (weaker)" if hit.get("weaker_match") else ""
        print(f"{i:2}. [{hit['kind']}] {hit['path']}{' — ' + place if place else ''}  "
              f"score {hit['score']} via {'+'.join(hit['matched_by'])}{weak}")
        if hit.get("excerpt"):
            print(f"      {hit['excerpt']}")
        print(f"      segment_id {hit['segment_id']}")


def doctor(service: DigUpService) -> int:
    ok = True

    def line(good: bool | None, text: str) -> None:
        nonlocal ok
        mark = {True: "ok  ", False: "FAIL", None: "warn"}[good]
        ok = ok and good is not False
        print(f"[{mark}] {text}")

    s = service.settings
    print(f"hermes-digup {__version__}")
    print(f"config file: {s.config_file or '(none; environment only)'}")
    try:
        snapshot = service.manager.get()
        counts = snapshot.counts()
        files = sum(sum(v.values()) for v in counts["files_by_kind"].values())
        line(True, f"index {snapshot.path}: {files} files, {counts['vectors']} vectors, schema {snapshot.schema}")
        for warning in snapshot.warnings:
            line(None, warning)
        age = service.manager.age_hours()
        if age is not None:
            line(age <= s.stale_hours or None, f"last synced {age:.1f} hours ago")
        source = snapshot.meta.get("vector_source")
        if source:
            print(f"       vectors made by: {source}")
    except IndexUnavailable as error:
        line(False, str(error))
        snapshot = None
    if service.manager.path(code=True).exists():
        try:
            code = service.manager.get(code=True)
            line(True, f"code index: {len(code.vectors)} vectors")
        except IndexUnavailable as error:
            line(False, f"code index: {error}")

    if not s.embed_urls:
        line(None, "DIGUP_EMBED_URL isn't set: searches will match exact words only")
    else:
        health = service.embedder.health()
        for result in health:
            if result["ok"]:
                line(True, f"embedding server {result['url']} answered in {result['ms']} ms")
                if result["model"] != s.embed_model:
                    line(None, f"it has no model {s.embed_model!r}, so its {result['model']!r} is used. "
                               f"Put DIGUP_EMBED_MODEL={result['model']} in {s.config_file or 'config.env'} "
                               "to make that the setting")
            else:
                line(False, f"embedding server {result['url']}: {result['error']}")
        if snapshot is not None and any(r["ok"] for r in health):
            try:
                check = service.fidelity()
                if check["checked"]:
                    good = {"match": True, "close": None, "mismatch": False}[check["verdict"]]
                    line(good, f"query vectors vs DigUp's: mean cosine {check['mean_cosine']} over "
                               f"{check['checked']} stored passages ({check['verdict']})")
                    if check["verdict"] != "match":
                        print("       The server isn't making the same vectors DigUp made. Serve Google's "
                              "google/embeddinggemma-2 as it is (not a re-quantized copy) with oMLX: "
                              "deploy/mac-server/setup-embedding-server.sh sets that up.")
                else:
                    print(f"       ({check['note']}; skipped the vector comparison)")
            except EmbeddingError as error:
                line(False, f"vector comparison: {error}")
    print("all good" if ok else "some checks failed")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-digup", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the MCP server (stdio by default, for Hermes)")
    serve.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    find = sub.add_parser("search", help="search from the shell")
    find.add_argument("query", nargs="+")
    find.add_argument("--kind", action="append", dest="kinds")
    find.add_argument("--folder")
    find.add_argument("--limit", type=int, default=10)
    find.add_argument("--code", action="store_true", help="search the code index")
    find.add_argument("--json", action="store_true")

    read = sub.add_parser("read", help="print the text of a segment_id or a file path")
    read.add_argument("target")
    read.add_argument("--code", action="store_true")

    sub.add_parser("status", help="index and embedding server status, as JSON")
    sub.add_parser("doctor", help="check the whole setup")

    args = parser.parse_args(argv)
    try:
        settings = Settings.load()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    # Never log to stdout: it's the MCP channel when serving over stdio.
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, settings.log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    service = DigUpService(settings)

    if args.command in (None, "serve"):
        from .server import build
        mcp = build(service)
        try:
            service.manager.get()  # load now, so the first search is quick
        except IndexUnavailable as error:
            logging.getLogger("hermes_digup").warning("%s", error)
        if getattr(args, "http", False):
            mcp.run("streamable-http", host=args.host, port=args.port)
        else:
            mcp.run("stdio")
        return 0
    try:
        if args.command == "search":
            _print(service.search(" ".join(args.query), kinds=args.kinds, limit=args.limit, folder=args.folder,
                                  code=args.code), args.json)
        elif args.command == "read":
            data = (service.read_passage(int(args.target), context=1, code=args.code) if args.target.isdigit()
                    else service.read_file(args.target, max_chars=60_000, code=args.code))
            print(data.get("text") or data.get("error") or data.get("note"))
        elif args.command == "status":
            print(json.dumps(service.status(), indent=2, ensure_ascii=False))
        elif args.command == "doctor":
            return doctor(service)
    except IndexUnavailable as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
