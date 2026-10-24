---
name: digup-search
description: Find and read the user's Mac files by what's in them.
version: 1.0.0
author: georgeslane, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [files, documents, search, semantic-search, pdf, photos, screenshots, mac, digup]
    category: productivity
---

# DigUp Search Skill

The user's files (documents, PDFs, photos, screenshots, recordings, videos, code) are on their Mac, not on the
machine Hermes runs on. The `digup` MCP server searches the index the DigUp app keeps of them, by meaning and by
exact words, and returns the text DigUp extracted. It's the only way to reach those files from here.

## When to Use

Use it **first**, before any other tool, whenever the user asks to find, look up, open, read, quote, summarise or
compare something in their own files, or asks "where is…", "do I have…", "what did … say", "my latest…":

- "the PDF where the landlord talks about pets", "my 2025 tax return", "that invoice from Acme"
- "the screenshot of the payment error", "photos of the dog on the beach"
- "what did that podcast say about sleep, and when?"
- "the function that retries with backoff" (code search, if the user turned it on in DigUp)

Don't look for these files with `terminal`, `search_files` or `read_file`: this machine's filesystem doesn't have
them, so searching it only wastes time. Use those tools only for files the user says are on this machine.

## Quick Reference

| Need | Tool |
|---|---|
| Find files by content or exact words (names, invoice numbers, error codes) | `mcp_digup_search_files` |
| Find code by what it does | `mcp_digup_search_code` |
| Read the matched page or passage (optionally with neighbours) | `mcp_digup_read_passage` |
| Read all the text of one file, in pages | `mcp_digup_read_file` |
| List files, newest first, by folder, kind or name | `mcp_digup_browse_files` |
| Check how fresh the index is | `mcp_digup_index_status` |

If these tools aren't listed directly, they're behind Tool Search: call them through `tool_describe` and
`tool_call` by the names above, without searching for them first.

## Procedure

1. Call `mcp_digup_search_files` with the user's description in plain words, the way they said it. Narrow with
   `kinds` (`pdf`, `doc`, `image`, `screenshot`, `audio`, `video`) or `folder` (`~/Documents/Taxes` means the folder
   on the Mac) when the user gives a hint.
2. Before quoting, summarising or answering from a file, read it: `mcp_digup_read_passage` with the result's
   `segment_id` (`context` 1–3 for the surrounding pages), or `mcp_digup_read_file` with its `path` for the whole
   text, following `next_offset`.
3. Answer with the file's name and Mac path, and where the match is (`page`, or `timestamp` for audio and video), so
   the user can open it on their Mac.
4. Nothing found: try once more with different words or fewer of them, or the exact words in the file, then say
   so. For "latest" or "in this folder" questions, `mcp_digup_browse_files` is quicker than a search.

## Pitfalls

- Paths are paths on the user's Mac. They don't exist on this machine: don't `read_file` them, use the digup tools.
- Pictures, audio and video are found by meaning; DigUp keeps no text for them (except screenshots' text). Describe
  what they show or say, and don't promise a transcript.
- `"mode": "keywords only"` in a result means the embedding server didn't answer: only exact words matched. Say so
  if it matters, and suggest exact words.
- A note that the index is old means files added since the last sync aren't in it yet (the Mac was asleep or away).
- Results marked `weaker_match` trail far behind the top ones; don't present them as answers.

## Verification

The answer names files that the search returned, and every quote or summary comes from text read with
`mcp_digup_read_passage` or `mcp_digup_read_file`.
