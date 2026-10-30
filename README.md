# Hermes-indexing-mcp-server

An MCP server that lets your [Hermes Agent](https://github.com/NousResearch/hermes-agent) search the files on
your Mac by what's in them. It reuses the index the [DigUp](https://github.com/ARahim3/DigUp) app already keeps,
and gets query embeddings from EmbeddingGemma 2, served by [oMLX](https://github.com/jundot/omlx) on a second,
always-on Mac.

Nothing is indexed twice. DigUp keeps indexing on your laptop the way it does now. This project copies its index to
the Pi on a schedule, and searches it there with the same ranking DigUp uses.

```
 Mac no.1 (laptop)                    Raspberry Pi                          Mac no.2 (always on)
 ┌──────────────────────┐            ┌───────────────────────────┐          ┌─────────────────────────┐
 │ DigUp app            │  rsync     │ Hermes Agent (gateway)    │  HTTP    │ oMLX                    │
 │  └ index.sqlite ─────┼──over ────▶│  └ hermes-digup (MCP)     │──query──▶│  EmbeddingGemma 2       │
 │ digup-sync (launchd) │  SSH,      │     ├ index copy          │◀─vector──│  (google's weights,     │
 │                      │  every     │     ├ meaning + keywords  │          │   pinned in memory)     │
 │ Hermes desktop app ──┼── 15 min   │     └ tools for the agent │          │  its own sub key        │
 │  (remote gateway) ───┼───────────▶│                           │          └─────────────────────────┘
 └──────────────────────┘ Tailscale  └───────────────────────────┘ Tailscale
```

## Contents

- [How it works](#how-it-works)
- [What you need](#what-you-need)
- [Setup](#setup)
  - [1. Tailscale names](#1-tailscale-names)
  - [2. Mac no.2: the embedding server](#2-mac-no2-the-embedding-server)
  - [3. The Pi: install the MCP server](#3-the-pi-install-the-mcp-server)
  - [4. Mac no.1: sync DigUp's index to the Pi](#4-mac-no1-sync-digups-index-to-the-pi)
  - [5. Add it to Hermes](#5-add-it-to-hermes)
  - [6. Check it](#6-check-it)
- [What Hermes can do with it](#what-hermes-can-do-with-it)
- [Making Hermes search here first](#making-hermes-search-here-first)
- [Running the embeddings on the Pi instead](#running-the-embeddings-on-the-pi-instead)
- [When something is down](#when-something-is-down)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)
- [Other setups](#other-setups)
- [Updating and uninstalling](#updating-and-uninstalling)
- [Development](#development)

## How it works

**The index.** DigUp keeps one SQLite file, `~/Library/Application Support/DigUp/index.noindex/index.sqlite`, plus
`code.sqlite` beside it if code search is on. The file holds every indexed file, a 768-dimension EmbeddingGemma 2
vector for every photo, PDF page, document passage, video frame and 30 seconds of audio, and a full-text keyword
index. This server reads that format directly (DigUp's schema 5) and never writes to it.

**The sync.** DigUp writes to its index while it runs, so copying the file as-is could catch it halfway through a
write. A launchd job on the laptop runs `digup-sync.sh` every 15 minutes. It:

1. takes a consistent snapshot with SQLite's backup API, including changes still in DigUp's write-ahead log;
2. runs an integrity check on the snapshot and turns it into one self-contained file;
3. sends it to the Pi with rsync over SSH. Only changed blocks cross the network, and rsync renames the finished file
   into place, so the Pi never sees half a file;
4. writes `manifest.json` (when and from where), which lets the Pi tell how fresh its copy is.

If the laptop is asleep, nothing runs. launchd runs a missed sync when the laptop wakes. If the Pi can't be reached,
the job logs it and tries again next time. If the files haven't changed, nothing is uploaded and only the "last
confirmed" time is updated.

**The search.** On the Pi, `hermes-digup` loads the vectors into memory and watches the index file. When a new copy
lands, it loads and checks the new copy first, then switches to it. If the new copy won't open, the old one keeps
answering and `index_status` reports why the new one was refused. Each query goes to Mac no.2 to be embedded with
DigUp's prompt (`task: search result | query: …`). The ranking is DigUp's own, ported from its Swift code:

- every vector is scored, with scores centred for each modality (text, image, audio);
- a file has to stand out by meaning (z ≥ 2) or contain at least half of the query's words;
- word matches add a bonus, which is why exact invoice numbers, names and error codes still work;
- words that name a kind of file ("…in a video", "screenshot of…") give a small boost.

**The embeddings.** Query vectors only make sense if they come from the same model DigUp used for the index. DigUp
runs an 8-bit llama.cpp build of EmbeddingGemma 2, and it checked that build against mlx-vlm running Google's own
weights (`google/embeddinggemma-2`): cosine 0.999 or better on text. oMLX loads EmbeddingGemma 2 through that same
mlx-vlm, so the setup script on Mac no.2 serves Google's weights with your oMLX. oMLX adds no prompt of its own; the
Pi adds DigUp's. `hermes-digup doctor` proves the match: it re-embeds passages from your index and compares them with
the vectors DigUp stored. A cosine of about 1.0 means the two are identical.

## What you need

| Machine | Needs |
|---|---|
| **Mac no.1** (laptop) | DigUp installed, with its first indexing pass done. Stock macOS tools only: bash, sqlite3, rsync, ssh. |
| **Mac no.2** (server) | Apple Silicon, [oMLX](https://github.com/jundot/omlx) (the menu bar app or the Homebrew install) with a main API key set, about 1.5 GB free disk for the model and about 1 GB of memory to keep it loaded. |
| **Raspberry Pi** | Hermes Agent, Raspberry Pi OS Bookworm or newer (Python 3.10+), rsync, and room for about two copies of the index (DigUp shows its size in Settings). A Pi 4 or 5 with 2 GB+ RAM. |
| **All three** | [Tailscale](https://tailscale.com), on the same tailnet. |

Clone this repository on the Pi and on both Macs (or copy the `deploy/` folder to the Macs):

```sh
git clone https://github.com/georgeslane/hermes-indexing-mcp-server.git
cd hermes-indexing-mcp-server
```

## Setup

Do the steps in this order: the Pi needs oMLX's address, key and model name from step 2, and the laptop needs the
Pi set up before it can sync.

### 1. Tailscale names

Find each machine's Tailscale name (MagicDNS) with `tailscale status` or in the Tailscale admin console. This guide
uses these example names, so swap in yours:

- `macbook`: Mac no.1
- `mac-mini`: Mac no.2
- `raspberrypi`: the Pi, with the user `pi`

If MagicDNS is off, use the `100.x.y.z` Tailscale IPs instead. Both work everywhere a name is asked for.

### 2. Mac no.2: the embedding server

oMLX needs a main API key before it can be reached from another machine. If you haven't set one, set it in the menu
bar app (Settings → Security) or in the web admin at `http://127.0.0.1:8000/admin`. Then, on the always-on Mac:

```sh
deploy/mac-server/setup-embedding-server.sh --expose
```

The script only uses oMLX's admin API, so your other models and settings stay as they are. It:

- finds your oMLX: its port and main key from `~/.omlx/settings.json` (or the folder the app uses), and starts it
  if it isn't running;
- with `--expose`, makes oMLX listen on the network (`0.0.0.0`) instead of only on this Mac, and restarts it. oMLX
  then refuses every request that has no key. Leave `--expose` out if oMLX already listens on the network;
- downloads `google/embeddinggemma-2` with oMLX's own downloader, unless oMLX already has it;
- **pins** the model, so oMLX keeps it loaded (text only, about 0.9 GB) and never unloads it for being idle. The first
  search after a quiet night is as quick as any other;
- creates an oMLX **sub key** named `hermes-digup` for the Pi. A sub key can run models but can't change oMLX's
  settings, and you can revoke it on its own. It's saved in `~/.config/hermes-digup/embedding-server.env`, and a
  later run reuses it;
- embeds a test query the way the Pi will, and checks that the vector has 768 dimensions;
- prints the `DIGUP_EMBED_URL`, `DIGUP_EMBED_API_KEY` and `DIGUP_EMBED_MODEL` lines for the Pi. Keep them for step 3.

Options:

- `--admin-key KEY`: if the main key isn't in oMLX's settings file (for example, you set it with `OMLX_API_KEY`).
- `--port 8000`: if the script can't tell which port oMLX uses.
- `--hf-token TOKEN`: if Hugging Face asks you to accept the model's license first.
- `--model-id NAME`: to use an EmbeddingGemma 2 that oMLX already has under another name.
- `--uninstall`: deletes the Pi's sub key and unpins the model.

Don't quantize EmbeddingGemma 2 with oQ for this. A re-quantized copy can drift away from DigUp's vectors; `doctor`
would show it.

Keep this Mac available:

- **Never sleep:** System Settings → Energy → turn on "Prevent automatic sleeping when the display is off". Or run
  `sudo pmset -a sleep 0`.
- **oMLX after a restart:** the menu bar app starts its server at launch, so add oMLX to System Settings → General →
  Login Items, and turn on automatic login (Users & Groups) if the Mac runs without a screen. With Homebrew,
  `brew services start omlx` does the same. Both restart oMLX if it crashes.
- **Firewall:** if the macOS firewall is on, allow oMLX the first time macOS asks.

From the Pi, check that you can reach it:

```sh
curl -s http://mac-mini:8000/health                       # {"status":"healthy",...}
```

### 3. The Pi: install the MCP server

On the Pi, with the values step 2 printed:

```sh
deploy/pi/install.sh --embed-url http://mac-mini:8000,http://100.x.y.z:8000 \
  --api-key <key from step 2> --model <model name from step 2>
```

oMLX names a model after its folder, so the name depends on how the model was downloaded: for example
`embeddinggemma-2`, or `embeddinggemma-2-bf16` for a community bf16 copy. If the name is wrong anyway, the Pi asks
oMLX for its model list and uses the EmbeddingGemma 2 it finds there (unquantized first). `doctor` then says which
name it used.

`DIGUP_EMBED_URL` takes several addresses, comma-separated, tried in order. Giving the MagicDNS name and the IP means
a DNS hiccup doesn't stop meaning search.

The script:

- checks Python, and that this Python's SQLite can read DigUp's keyword index (FTS5 with DigUp's tokenizer);
- installs `python3-venv` and `rsync` with apt if they're missing;
- installs the server into its own virtualenv at `~/.local/share/hermes-digup/venv` and links the command to
  `~/.local/bin/hermes-digup`;
- creates `~/.local/share/hermes-digup/index/`, where the laptop will put the index;
- writes `~/.config/hermes-digup/config.env`. The MCP server and `hermes-digup doctor` both read this file, so the
  agent and your shell always use the same settings;
- installs the `digup-search` Hermes skill into `~/.hermes/skills/productivity/digup-search/` (see
  [Making Hermes search here first](#making-hermes-search-here-first)). If the desktop app uses a profile other than
  the default one, add `--profile NAME`.

Make sure SSH is on, so the laptop can log in to copy files: `sudo raspi-config` → Interface Options → SSH.

### 4. Mac no.1: sync DigUp's index to the Pi

On the laptop that runs DigUp, first give it a passwordless SSH key for the Pi. launchd can't type passwords:

```sh
ssh-keygen -t ed25519 -f ~/.ssh/hermes_digup -N ""
ssh-copy-id -i ~/.ssh/hermes_digup.pub pi@raspberrypi
```

Then install the sync job:

```sh
deploy/mac-laptop/install.sh --pi pi@raspberrypi --key ~/.ssh/hermes_digup
```

The script:

- copies `digup-sync.sh` to `~/Library/Application Support/hermes-digup/`, so the job keeps working even if you
  move or delete the repository;
- writes `~/.config/hermes-digup/sync.env`;
- checks that SSH to the Pi works without a password and that the Pi has rsync;
- installs the launchd job `com.hermes-digup.sync`, which runs at login, then every 15 minutes, and after a wake
  when a run was missed. The first sync starts right away.

`--interval 600` changes how often it syncs, in seconds. Ten to thirty minutes is a sensible range. DigUp makes a new
file searchable within seconds, so a shorter interval means fresher results on the Pi. Each run is cheap: an
unchanged index sends nothing, and a changed one sends only the changed blocks.

The sync finds the index on its own. It looks at `DIGUP_INDEX_DIR` in `sync.env`, then at DigUp's own "index folder"
setting, then at DigUp's default location.

To run a sync by hand and watch it:

```sh
~/Library/Application\ Support/hermes-digup/digup-sync.sh --force -v
tail -f ~/Library/Logs/hermes-digup-sync.log
```

> Tailscale SSH (`tailscale set --ssh` on the Pi) also works with no keys at all. But the default tailnet policy
> asks you to sign in again in a browser every 12 hours ("check" mode), which breaks unattended syncs. Plain SSH keys
> are the reliable choice, unless your policy uses `"action": "accept"`.

### 5. Add it to Hermes

Hermes runs on the Pi, so the MCP server runs there too, as a subprocess of Hermes. On the Pi, run:

```sh
hermes mcp add digup --command ~/.local/bin/hermes-digup --args serve
```

or add this to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  digup:
    command: "/home/pi/.local/bin/hermes-digup"   # the path install.sh printed
    args: ["serve"]
    timeout: 60
    connect_timeout: 30
```

Then run `/reload-mcp` in a Hermes chat. A running gateway notices the config change by itself within about a
minute. You can also add the server from the Hermes desktop app on the laptop: **Settings → MCP servers**, with your
Pi's gateway selected. MCP setup always applies to the gateway the app is connected to, which here is the Pi.

All settings come from `~/.config/hermes-digup/config.env`, so this block needs no `env:`. Any setting can still be
overridden there; for example, `env: {DIGUP_STALE_HOURS: "12"}`.

Hermes registers the tools as `mcp_digup_search_files`, `mcp_digup_read_passage` and so on. Every tool is marked
read-only.

### 6. Check it

On the Pi:

```sh
hermes-digup doctor
```

```
[ok  ] index /home/pi/.local/share/hermes-digup/index/index.sqlite: 18342 files, 61877 vectors, schema 5
[ok  ] last synced 0.2 hours ago
       vectors made by: google/embeddinggemma-2@bfcd298 llama.cpp-vectors1 text-q8_0+mmproj-q8_0
[ok  ] embedding server http://mac-mini:8000 answered in 38 ms
[ok  ] query vectors vs DigUp's: mean cosine 0.9996 over 6 stored passages (match)
all good
```

Then search from the shell:

```sh
hermes-digup search "the clause about pets in the lease"
hermes-digup search "screenshot of the payment declined error" --kind screenshot
hermes-digup search "INV-2026-0042"
hermes-digup read 48213          # the text of a result's segment_id
```

Then ask Hermes, from the desktop app on your laptop, something like *"find the PDF where my landlord talks about
pets"*, *"what did that podcast say about sleep, and when?"* or *"summarise the latest bank statement in
Downloads"*.

## What Hermes can do with it

| Tool | What it does |
|---|---|
| `search_files` | Hybrid search (meaning plus exact words). Takes optional `kinds` (screenshot, image, pdf, doc, audio, video), `folder` (`~/Documents/Taxes` means that folder on the Mac) and `limit`. Each result gives the path on the Mac, kind, score, where the match is (`page`, a `timestamp` for audio/video, `chunk`, `lines`), an excerpt with matched words marked «like this», the date modified, and a `segment_id`. |
| `search_code` | The same over DigUp's code index ("retry a failed call with backoff"), if code search is on in DigUp. Uses DigUp's code-retrieval prompt. |
| `read_passage` | The full text of a match (a PDF page, a document passage, a screenshot's text, a stretch of code), with up to 3 neighbouring passages on each side. |
| `read_file` | All the text DigUp extracted from one file, in order, in pages (`offset`, `next_offset`). Lets the agent read or summarise a PDF or document even though the file itself is on the laptop. |
| `browse_files` | Lists indexed files, newest first, filtered by folder, kind or part of the name ("my latest screenshots"). |
| `index_status` | How fresh the copy is, what's in it, and whether the embedding server answers. |

Pictures, audio and video are found by meaning. DigUp keeps vectors for them, not words, so `read_*` has no text for
them, except for screenshots, whose text DigUp reads on a Mac. Paths are paths on the laptop, which you can open from
there.

## Making Hermes search here first

Out of the box Hermes can take a while to reach for these tools. It tends to search the Pi's own disk first with its
built-in file tools, which can never find your Mac's files. Two things cause this:

- **Tool Search.** Hermes's Tool Search is on by default. It keeps every MCP tool out of the model's tool list,
  showing only a one-line preview of each, while its own file tools are always loaded.
- **What the model knows.** Nothing tells the model that your files live on another machine.

These settings fix both. They're in order of effect; the first two together are usually enough.

**1. Pin the `digup-search` skill** (`deploy/pi/install.sh` installs it). Its full text then goes into the system
prompt of every new chat. It tells the model that your files are on your Mac, to use the digup tools first and not
search this machine's disk, and how to read a match before quoting it. In the Pi's `~/.hermes/config.yaml` (or the
profile's), add:

```yaml
skills:
  auto_load:
    - digup-search
```

It costs about 1,000 tokens per chat. Without pinning, the skill still appears in Hermes's skill list and the model is
told to load matching skills, so it helps either way; pinned, it's certain.

**2. Tell Hermes to remember it.** Memory is also part of every chat's system prompt. In any chat, say:

> Remember: my files are on my Mac, not on this machine. To find, read or summarise any of my files, use the digup
> tools first, never the local filesystem or terminal.

**3. Load the tools directly** (optional). If digup is your only MCP server, or you have just a few, turn Tool Search
off so their full descriptions are always in front of the model:

```yaml
tools:
  tool_search:
    enabled: off
```

This loads every MCP server's tools into every chat, so skip it if you have large servers attached. Even with Tool
Search on, each tool's preview starts by saying what it's for (for example, "Find the user's Mac files by content;
use it first.").

Changes apply to **new** chats: start one in the desktop app after editing.

## When something is down

| Situation | What happens |
|---|---|
| Laptop asleep, off, or away | The Pi keeps searching its last copy. When the copy is older than `DIGUP_STALE_HOURS` (default 36), results include a note saying so. The laptop syncs when it wakes. |
| Pi unreachable during a sync | The sync logs it and exits. The next run sends everything that changed. |
| Sync killed halfway (laptop sleeps, network drops) | rsync never put a partial file in place. The Pi still has the previous complete copy, and the next run finishes the job. A leftover lock is cleared automatically. |
| A broken or unexpected file arrives | The new copy is checked before it's used. If it fails, the old one keeps answering and `index_status` shows `newer_copy_refused` with the reason. The sync also runs an integrity check before uploading anything. |
| Mac no.2 asleep, rebooting, or unreachable | Each embedding address gets a 3-second connection check, then the next address is tried. If none answers, the search runs on exact words only, and the result says `"mode": "keywords only"` with a note. Meaning search comes back on its own. |
| The embedding server returns the wrong kind of vector (wrong model, wrong size, token vectors) | A clear error saying what's wrong. `doctor`'s comparison catches a model that's merely different. |
| DigUp turns code search off | The next sync removes `code.sqlite` from the Pi too. |
| DigUp is updated and its index schema changes | Known schemas load normally. A newer one is tried anyway, with a warning in results and status asking you to update hermes-digup. |
| The same question twice | Its vector is cached (the last 512 queries), so Mac no.2 isn't asked again. |

## Settings

**Pi: `~/.config/hermes-digup/config.env`.** Environment variables override this file, and `DIGUP_CONFIG` points to
a different one.

| Setting | Default | Meaning |
|---|---|---|
| `DIGUP_INDEX_DIR` | `~/.local/share/hermes-digup/index` | Folder holding `index.sqlite`, `code.sqlite` and `manifest.json`. |
| `DIGUP_EMBED_URL` | (none) | Embedding server base URL(s), comma-separated, tried in order. Without one, only keyword search works. |
| `DIGUP_EMBED_API_KEY` | (none) | Bearer token for the embedding server. |
| `DIGUP_EMBED_API` | `openai` | `openai` for oMLX (and llama-server, LM Studio, or Ollama's `/v1`); `ollama` for Ollama's `/api/embed`. |
| `DIGUP_EMBED_MODEL` | `embeddinggemma-2` | The model's name in oMLX, as the setup script printed it. If the server has no model by that name, its EmbeddingGemma 2 is used, whatever it's called. |
| `DIGUP_EMBED_TIMEOUT` | `15` | Seconds to wait for a vector once connected. |
| `DIGUP_STALE_HOURS` | `36` | Age after which results mention that the index is old. |
| `DIGUP_FLOAT32` | auto | `1` keeps vectors as float32 (about 4× faster scoring, twice the memory), `0` keeps float16. Auto picks float32 when it fits in a quarter of free memory. |
| `DIGUP_LOG_LEVEL` | `INFO` | Logs go to stderr, which Hermes saves in `~/.hermes/logs/mcp-stderr.log`. |

**Laptop: `~/.config/hermes-digup/sync.env`**

| Setting | Default | Meaning |
|---|---|---|
| `PI_HOST` | (required) | `user@host` for SSH. |
| `PI_INDEX_DIR` | `.local/share/hermes-digup/index` | Where the copy goes on the Pi, relative to the Pi user's home. Must match `DIGUP_INDEX_DIR`. |
| `SSH_KEY` | (none) | Private key to use (a path without spaces). |
| `DIGUP_INDEX_DIR` | auto | Only if DigUp's index lives somewhere unusual. |
| `SKIP_ON_BATTERY` | `0` | `1` skips syncing on battery. |
| `SYNC_CODE_INDEX` | `1` | Also copy `code.sqlite` when DigUp's code search is on. |

To change the sync interval, run `install.sh --interval <seconds>` again.

## Troubleshooting

**`hermes-digup doctor` says there's no index.** The laptop hasn't synced yet. Run
`~/Library/Application\ Support/hermes-digup/digup-sync.sh --force -v` on it and read the output. The usual causes:

- SSH asks for a password: set up the key from step 4.
- rsync is missing on the Pi: `sudo apt install rsync`.
- DigUp hasn't finished its first pass yet.

**"the Pi isn't reachable" in the sync log.** Check that `ssh pi@raspberrypi` works from the laptop and that
Tailscale is connected on both machines.

**The sync log says it can't open the index ("operation not permitted").** You moved DigUp's index into a folder
macOS protects, such as Documents or Desktop. Give `/bin/bash` Full Disk Access (System Settings → Privacy &
Security), or keep the index in its default place.

**The comparison says "mismatch" (cosine well below 1).** oMLX isn't making DigUp's vectors. Check that
`DIGUP_EMBED_MODEL` names Google's `google/embeddinggemma-2` as oMLX downloaded it, not an oQ-quantized or
community-converted copy. Re-running `setup-embedding-server.sh` sets this up. A "close" result (0.9 to 0.98) still
searches, but ranks a little worse than DigUp.

**Searches say "keywords only".** Mac no.2 didn't answer. Check that oMLX is running there (its menu bar icon, or
`brew services info omlx`), and look at `~/.omlx/logs/server.log`. From the Pi, run
`curl http://mac-mini:8000/health`. If it only answers on Mac no.2 itself, oMLX listens on 127.0.0.1: run the setup
script again with `--expose`.

**HTTP 401.** `DIGUP_EMBED_API_KEY` on the Pi doesn't match the sub key in Mac no.2's
`~/.config/hermes-digup/embedding-server.env`, or the sub key was deleted in oMLX. Re-running the setup script makes
a new one.

**"no model 'embeddinggemma-2'" or "It has no EmbeddingGemma 2 model".** When `DIGUP_EMBED_MODEL` isn't the model's
name in oMLX, the Pi uses the server's EmbeddingGemma 2 under its own name, and `doctor` shows a `[warn]` line with the
setting to put in `config.env`. If oMLX has no EmbeddingGemma 2 at all, the error lists the models it has: run the
setup script on Mac no.2 to download it.

**Hermes doesn't show the tools.** Run `hermes mcp test digup`, check `~/.hermes/logs/mcp-stderr.log`, and make sure
the `command:` path exists (`ls ~/.local/bin/hermes-digup`). Run `/reload-mcp` after changes.

**A result is missing.** First check that DigUp itself finds it on the laptop. If it does, the Pi's copy is just
older: run `index_status`, or `hermes-digup status`, to see when it was last synced.

## Running the embeddings on the Pi instead

The Pi only embeds search queries, so it can run EmbeddingGemma 2 itself and free the ~1 GB oMLX keeps loaded on Mac
no.2. Your index doesn't change and nothing is re-indexed. The Pi runs the very file DigUp made the index with:
`embeddinggemma-2-Q8_0.gguf` at DigUp's pinned revision, checksum-verified. It's run by the same llama.cpp release
DigUp builds (b11461, with DigUp's one-line memory patch) and mean-pooled the same way. That's an even closer match to
your index than oMLX's bf16 copy.

On the Pi (a Pi 5, or a Pi 4 with 4 GB or more):

```sh
deploy/pi/setup-embedding-server.sh
```

The script:

- installs `git`, `cmake` and a compiler with apt if they're missing;
- downloads llama.cpp at b11461, checks it's the exact commit DigUp pins, applies DigUp's patch, and builds only
  `llama-server`, optimized for this CPU. This happens once and takes about 10 to 20 minutes on a Pi 5;
- downloads the text part of the model (310 MB, resumable) and checks DigUp's SHA-256;
- runs it as the systemd user service `hermes-digup-embeddings`. It listens on `127.0.0.1:8088` only, so nothing else
  on your network can reach it and it needs no key. systemd restarts it if it stops, and lingering is turned on so it
  starts at boot;
- embeds a test query and checks for a 768-dimension vector;
- puts `http://127.0.0.1:8088` first in `DIGUP_EMBED_URL` and keeps the Mac's addresses after it, so the Mac is used
  only if the Pi's server is down;
- runs `hermes-digup doctor`. Its comparison line should say "match".

Then:

1. **Restart the MCP server** with `/reload-mcp` in Hermes, or restart Hermes. The MCP server reads `DIGUP_EMBED_URL`
   when it starts.
2. **Free the Mac's memory.** On Mac no.2, run `deploy/mac-server/setup-embedding-server.sh --unpin`. This unpins and
   unloads EmbeddingGemma 2 but keeps the Pi's key, so the Mac still answers as the fallback. oMLX then loads the model
   only if the Pi's server is down.

What it costs the Pi:

- **Memory:** about 0.5 GB while serving. The service also asks systemd to cap it at 1.5 GB, so it can't crowd out
  Hermes, where your system enforces memory limits on user services (Raspberry Pi OS normally does).
- **Speed:** a query is short, so expect well under a second on a Pi 5. The test query's time is printed at the end.
  Repeated queries are cached anyway.

Options: `--port`, `--threads` (all cores by default), and `--no-fallback` to drop the Mac from `DIGUP_EMBED_URL`.
`--uninstall` stops the service and puts the Mac first again. After that, run the Mac's script without `--unpin` to pin
the model there again.

Check on it with `systemctl --user status hermes-digup-embeddings` and `journalctl --user -u hermes-digup-embeddings`.

## Other setups

**Ollama, LM Studio or llama.cpp on Mac no.2 instead of oMLX.** Set `DIGUP_EMBED_URL` to the server and
`DIGUP_EMBED_MODEL` to its name for EmbeddingGemma 2. With llama.cpp, the closest match is the file DigUp itself
runs: `ggml-org/embeddinggemma-2-GGUF`'s `embeddinggemma-2-Q8_0.gguf`, served by
`llama-server --embedding --pooling mean -c 2048 -ub 2048`. For Ollama, either set `DIGUP_EMBED_API=ollama` (its native
`/api/embed`) or keep `openai` with the URL ending in `/v1`. Then run `hermes-digup doctor` and look at the
comparison line. Only a "match" means search will rank like DigUp.

**Running the server on the laptop itself.** Point `DIGUP_INDEX_DIR` at DigUp's own folder
(`~/Library/Application Support/DigUp/index.noindex`). The server sees DigUp's write-ahead log, opens the live index
read-only next to DigUp, and picks up DigUp's changes as they happen, with no sync needed. This is handy for trying
things out, or if Hermes ever runs on the Mac.

**Hermes reaching the server over HTTP.** Run `hermes-digup serve --http --host 100.x.y.z --port 8765` (for example
under systemd) and use `url: "http://100.x.y.z:8765/mcp"` in Hermes's config instead of `command:`.

## Updating and uninstalling

- **Update:** `git pull`, then on the Pi run `deploy/pi/install.sh` again (it keeps your config). On the Macs, run
  their scripts again.
- **Pi:** remove the `digup` entry from Hermes, then `rm -rf ~/.local/share/hermes-digup ~/.config/hermes-digup
  ~/.local/bin/hermes-digup`.
- **Laptop:** `deploy/mac-laptop/install.sh --uninstall`.
- **Mac no.2:** `deploy/mac-server/setup-embedding-server.sh --uninstall` deletes the Pi's sub key and unpins the
  model. To free its disk space too, delete EmbeddingGemma 2 on oMLX's Models page.

## Development

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

The tests build indexes with DigUp's exact schema and run the real embedding client against a fake embedding
server. They cover:

- meaning, keyword and lookup ranking;
- filters;
- reading pages and documents (chunk overlaps merged);
- hot reload, and refusing a broken copy;
- a live WAL-mode index;
- failover, timeouts and the keyword fallback;
- Ollama and API keys;
- the vector comparison;
- the sync script, against a local stand-in for the Pi;
- the Pi's embedding-server script, with stand-ins for git, cmake, the download and systemd, and a fake
  `llama-server` that systemd really starts;
- the oMLX setup script, against a fake oMLX (its JavaScript JSON helper runs under Node instead of `osascript`);
- the Hermes skill: Hermes's authoring rules, that it names exactly the server's tools, and that every tool's
  preview fits Tool Search's 60 characters.

Layout:

```
src/hermes_digup/
  index.py     reads DigUp's SQLite index, validates and hot-reloads snapshots
  search.py    DigUp's hybrid ranking (Searcher.swift), ported
  textnorm.py  DigUp's query normalisation, stopwords and kind words (SearchText.swift)
  embed.py     EmbeddingGemma 2 client: DigUp's prompts, failover, cache
  service.py   the tools' logic; server.py exposes it over MCP; cli.py the shell commands
hermes/skills/
  digup-search/ the Hermes skill that makes the agent search the Mac's files here first
deploy/
  mac-laptop/  digup-sync.sh + launchd installer (Mac no.1)
  mac-server/  oMLX setup for EmbeddingGemma 2, through its admin API (Mac no.2)
  pi/          virtualenv installer and Hermes config, and the optional on-Pi embedding server (Raspberry Pi)
```

DigUp's index format isn't a public API, so a DigUp update could change it. The reader checks the schema version and
tells you when it sees one it doesn't know.

## License

MIT. DigUp is MIT licensed, EmbeddingGemma 2 is Apache 2.0, and oMLX is Apache 2.0. This project isn't affiliated
with DigUp, oMLX, Google or Nous Research.
