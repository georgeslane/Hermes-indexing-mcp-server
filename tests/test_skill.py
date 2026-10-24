"""The digup-search skill follows Hermes's authoring rules (its tests/skills/test_authoring_standards.py) and
names only tools this server has."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from hermes_digup.config import Settings
from hermes_digup.server import build
from hermes_digup.service import DigUpService

SKILL = Path(__file__).resolve().parents[1] / "hermes" / "skills" / "digup-search" / "SKILL.md"


def frontmatter() -> tuple[dict[str, str], str]:
    content = SKILL.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    head, _, body = content[4:].partition("\n---\n")
    fields = dict(re.findall(r"^([a-z_]+): ?(.*)$", head, re.MULTILINE))
    return fields, body


def test_frontmatter_meets_hermes_rules():
    fields, _ = frontmatter()
    for name in ("name", "description", "version", "author", "license", "platforms"):
        assert name in fields, f"missing {name}"
    assert fields["name"] == SKILL.parent.name
    assert len(fields["description"]) <= 60 and fields["description"].endswith(".")
    assert "tags:" in SKILL.read_text(encoding="utf-8")


def test_names_only_real_tools():
    _, body = frontmatter()
    mcp = build(DigUpService(Settings.load({"DIGUP_CONFIG": "/nonexistent"})))
    tools = {tool.name for tool in asyncio.run(mcp.list_tools())}
    named = set(re.findall(r"`mcp_digup_([a-z_]+)`", body))
    assert named and named <= tools, named - tools
    assert tools <= named, f"the skill doesn't mention {tools - named}"


def test_tool_previews_fit_tool_search():
    """Tool Search shows each tool's first sentence, cut at 60 characters."""
    mcp = build(DigUpService(Settings.load({"DIGUP_CONFIG": "/nonexistent"})))
    for tool in asyncio.run(mcp.list_tools()):
        first = re.split(r"(?<=[.!?;])\s", tool.description.strip(), maxsplit=1)[0]
        assert len(first) <= 60, f"{tool.name}: {first!r}"
