"""The MCP server Hermes talks to."""

from __future__ import annotations

from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from .index import IndexUnavailable
from .service import DigUpService

Kind = Literal["screenshot", "image", "pdf", "doc", "audio", "video"]

INSTRUCTIONS = """\
Search the files on the user's Mac by what is in them, using the index the DigUp app keeps there.
Use these tools first for any question about the user's files: they're on the user's Mac, not on the
machine this server runs on, so the local filesystem and terminal can't find them.
It covers photos and screenshots (by what they show), PDFs page by page, documents, spreadsheets,
slides, audio and video (by what is said and shown, with timestamps), and code if code search is on.
Describe what you're looking for in plain words, the way the user would ("the lease clause about
pets", "screenshot of the payment declined error", "where they talk about sleep"); exact names,
invoice numbers and error codes are matched too. Paths are paths on the user's Mac. Use
read_passage or read_file to read the matched text before quoting or summarising it."""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)


def build(service: DigUpService) -> MCPServer:
    mcp = MCPServer("digup", instructions=INSTRUCTIONS)

    def guarded(fn, *args, **kwargs) -> dict:
        try:
            return fn(*args, **kwargs)
        except IndexUnavailable as error:
            return {"error": str(error)}

    @mcp.tool(annotations=READ_ONLY)
    def search_files(
        query: Annotated[str, Field(description="What to find, described in plain words, or exact words/codes "
                                                "that are in the file.")],
        kinds: Annotated[list[Kind] | None, Field(description="Only these kinds of file.")] = None,
        folder: Annotated[str | None, Field(description="Only files under this folder on the Mac, e.g. "
                                                        "'~/Documents/Taxes'.")] = None,
        limit: Annotated[int, Field(ge=1, le=50, description="How many files to return.")] = 10,
    ) -> dict:
        """Find the user's Mac files by content; use it first.

        Searches DigUp's index of the user's Mac, by meaning and exact words: documents, PDFs page by page,
        photos and screenshots by what they show, audio and video by what's said, with timestamps. Use it
        first for any question about the user's files; they aren't on this machine, so the terminal and the
        local file tools can't find them. Returns the best matching files with where the match is (page,
        timestamp, chunk), an excerpt, and a segment_id for read_passage. Results marked weaker_match trail
        far behind the top ones."""
        return guarded(service.search, query, kinds=kinds, limit=limit, folder=folder)

    @mcp.tool(annotations=READ_ONLY)
    def search_code(
        query: Annotated[str, Field(description="What the code does ('retry a failed call with backoff') or a "
                                                "name that's in it.")],
        folder: Annotated[str | None, Field(description="Only files under this folder on the Mac.")] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> dict:
        """Find code in the user's Mac projects by what it does.

        Searches DigUp's code index (if code search is turned on in DigUp). Returns files with line ranges;
        read them with read_passage(code=true) or read_file(code=true)."""
        return guarded(service.search, query, limit=limit, folder=folder, code=True)

    @mcp.tool(annotations=READ_ONLY)
    def read_passage(
        segment_id: Annotated[int, Field(description="segment_id from a search result.")],
        context: Annotated[int, Field(ge=0, le=3, description="Also return this many neighbouring pages or "
                                                              "passages on each side.")] = 0,
        code: Annotated[bool, Field(description="True for a segment_id from search_code.")] = False,
    ) -> dict:
        """Read the text of a digup search match (segment_id).

        The full text of the matched page, passage or code stretch, as DigUp read it."""
        return guarded(service.read_passage, segment_id, context=context, code=code)

    @mcp.tool(annotations=READ_ONLY)
    def read_file(
        path: Annotated[str, Field(description="A path exactly as a search returned it.")],
        offset: Annotated[int, Field(ge=0, description="Character offset to continue from (next_offset).")] = 0,
        max_chars: Annotated[int, Field(ge=1000, le=60000)] = 20000,
        code: Annotated[bool, Field(description="True for a file from search_code.")] = False,
    ) -> dict:
        """Read all the text of one of the user's Mac files.

        Everything DigUp extracted from the file (PDF pages, document text, slide and sheet text, screenshot
        text), in order, in pages of max_chars. Use it for paths from a digup search, which don't exist on
        this machine. Pictures, audio and video have no text."""
        return guarded(service.read_file, path, offset=offset, max_chars=max_chars, code=code)

    @mcp.tool(annotations=READ_ONLY)
    def browse_files(
        folder: Annotated[str | None, Field(description="Only files under this folder on the Mac.")] = None,
        kind: Annotated[Kind | None, Field(description="Only this kind of file.")] = None,
        name_contains: Annotated[str | None, Field(description="Part of the file or folder name.")] = None,
        limit: Annotated[int, Field(ge=1, le=200)] = 20,
    ) -> dict:
        """List the user's Mac files, newest first.

        Filter by folder, kind or part of the name: 'my latest screenshots', 'PDFs in Downloads'."""
        return guarded(service.browse, folder=folder, kind=kind, name_contains=name_contains, limit=limit)

    @mcp.tool(annotations=READ_ONLY)
    def index_status() -> dict:
        """Check how fresh the user's Mac file index is.

        When it was last synced, what's in it, and whether the embedding server answers."""
        return service.status()

    return mcp
