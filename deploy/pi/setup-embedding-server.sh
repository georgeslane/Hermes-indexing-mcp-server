#!/bin/bash
# Runs EmbeddingGemma 2 on this Raspberry Pi for hermes-digup's searches, so Mac no.2 doesn't have to keep it in memory.
#
# No reindexing: it's the same model file DigUp made your index with (ggml-org's embeddinggemma-2-Q8_0.gguf at DigUp's
# pinned revision, checksum-verified), run by the same llama.cpp release DigUp builds (b11461, with DigUp's one-line
# memory patch), mean-pooled like DigUp. The Pi only embeds search queries, so it needs only the text part (310 MB).
#
#   1. builds llama-server from llama.cpp b11461 (once; about 10 to 20 minutes on a Pi 5);
#   2. downloads the model file and checks DigUp's SHA-256;
#   3. runs it as a systemd user service on 127.0.0.1 only (nothing else on the network can reach it, so no key);
#   4. puts it first in DIGUP_EMBED_URL in ~/.config/hermes-digup/config.env, keeping the Mac as the fallback;
#   5. runs `hermes-digup doctor`, whose vector comparison proves the vectors still match DigUp's.
#
#   deploy/pi/setup-embedding-server.sh [--port 8088] [--threads 4] [--no-fallback]
#   deploy/pi/setup-embedding-server.sh --uninstall     # stop the service and go back to the Mac
#
# --no-fallback   use only the Pi (drop the Mac's addresses from DIGUP_EMBED_URL)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$HOME/.local/share/hermes-digup"
SOURCE="$DATA/llama.cpp"
BUILD="$DATA/llama-build"
MODELS="$DATA/models"
CONFIG="$HOME/.config/hermes-digup/config.env"
SERVICE=hermes-digup-embeddings.service
UNIT_FILE="$HOME/.config/systemd/user/$SERVICE"

# The llama.cpp release DigUp builds its embedder from (DigUp's scripts/build-llama.sh), and its patch.
TAG=b11461
COMMIT=4d756bc72bf00a4aacf410ae15a2d315f3db400d   # TAG's commit: a tag can be moved, a commit can't
PATCH="$HERE/llama.cpp-no-logits.patch"
# DigUp's model file (Sources/DigUpKit/ModelFiles.swift): the text part is all that queries need.
REPO_ID="ggml-org/embeddinggemma-2-GGUF"
REVISION="bfcd298762cc34d0357ece5ebdd31791a3a374d8"
FILE="embeddinggemma-2-Q8_0.gguf"
SHA256="2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
BYTES=309855456

PORT=8088
THREADS="$(nproc 2>/dev/null || echo 4)"
FALLBACK=1
UNINSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --no-fallback) FALLBACK=0; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }
LOCAL_URL="http://127.0.0.1:$PORT"
[ "$(uname -s)" = Linux ] || die "this script is for the Raspberry Pi (Linux)"
[ -f "$CONFIG" ] || die "no $CONFIG: run deploy/pi/install.sh first"

setting() { sed -n "s/^$1=//p" "$CONFIG" | tail -1; }
# set_urls "a,b,c": DIGUP_EMBED_URL in config.env, as given, with duplicates and blanks dropped.
set_urls() {
  local list="" url
  IFS=',' read -r -a parts <<< "$1"
  for url in "${parts[@]}"; do
    url="$(echo "$url" | tr -d ' ')"
    url="${url%/}"
    [ -z "$url" ] && continue
    case ",$list," in *",$url,"*) continue ;; esac
    list="${list:+$list,}$url"
  done
  if grep -q '^DIGUP_EMBED_URL=' "$CONFIG"; then
    sed -i "s|^DIGUP_EMBED_URL=.*|DIGUP_EMBED_URL=$list|" "$CONFIG"
  else
    echo "DIGUP_EMBED_URL=$list" >> "$CONFIG"
  fi
  echo "DIGUP_EMBED_URL=$list"
}
# The configured addresses other than this Pi's own server.
other_urls() {
  local url out=""
  IFS=',' read -r -a parts <<< "$(setting DIGUP_EMBED_URL)"
  for url in "${parts[@]}"; do
    url="$(echo "$url" | tr -d ' ')"
    url="${url%/}"
    case "$url" in ""|"$LOCAL_URL"|"http://localhost:$PORT") ;; *) out="${out:+$out,}$url" ;; esac
  done
  echo "$out"
}

if [ "$UNINSTALL" = 1 ]; then
  systemctl --user disable --now "$SERVICE" 2>/dev/null || true
  rm -f "$UNIT_FILE"
  systemctl --user daemon-reload 2>/dev/null || true
  rest="$(other_urls)"
  set_urls "$rest" > /dev/null
  echo "Stopped and removed $SERVICE."
  if [ -n "$rest" ]; then
    echo "Searches go to the Mac again: DIGUP_EMBED_URL=$rest"
    echo "On Mac no.2, pin EmbeddingGemma 2 again: deploy/mac-server/setup-embedding-server.sh"
  else
    echo "warning: DIGUP_EMBED_URL is now empty, so searches match exact words only. Set it to the Mac's oMLX." >&2
  fi
  echo "The build and the model are still in $DATA (llama.cpp, llama-build, models): delete them to free ~1 GB."
  echo "Then reload the MCP server in Hermes: /reload-mcp"
  exit 0
fi

# 0. Room to run: about 0.5 GB while serving; the build wants more for a few minutes.
available_mb=$(( $(sed -n 's/^MemAvailable: *\([0-9]*\) kB/\1/p' /proc/meminfo 2>/dev/null || echo 0) / 1024 ))
if [ "$available_mb" -gt 0 ] && [ "$available_mb" -lt 1500 ]; then
  echo "warning: only ${available_mb} MB of memory is free; the model server needs about 500 MB." >&2
fi

# 1. Build tools.
missing=()
command -v git >/dev/null || missing+=(git)
command -v cmake >/dev/null || missing+=(cmake)
command -v g++ >/dev/null || missing+=(build-essential)
command -v curl >/dev/null || missing+=(curl)
command -v python3 >/dev/null || missing+=(python3)
if [ ${#missing[@]} -gt 0 ]; then
  echo "Installing ${missing[*]} (needs sudo)..."
  sudo apt-get update -qq && sudo apt-get install -y "${missing[@]}"
fi

# 2. llama.cpp at DigUp's release, with DigUp's patch.
mkdir -p "$DATA" "$MODELS"
if [ ! -d "$SOURCE/.git" ]; then
  echo "Fetching llama.cpp $TAG..."
  git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SOURCE"
elif [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$COMMIT" ]; then
  git -C "$SOURCE" fetch --quiet --depth 1 origin tag "$TAG"
  git -C "$SOURCE" checkout --quiet --force "$TAG"
fi
[ "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT" ] \
  || die "llama.cpp's $TAG isn't commit $COMMIT here: not building it"
if git -C "$SOURCE" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SOURCE" apply "$PATCH"
elif ! git -C "$SOURCE" apply --reverse --check "$PATCH" 2>/dev/null; then
  die "$PATCH doesn't apply to $SOURCE, and isn't applied already"
fi

SERVER_BIN="$BUILD/bin/llama-server"
STAMP="$TAG $(sha256sum "$PATCH" | cut -c1-12) $(uname -m)"
if [ ! -x "$SERVER_BIN" ] || [ "$(cat "$BUILD/STAMP" 2>/dev/null)" != "$STAMP" ]; then
  echo "Building llama-server (once; about 10 to 20 minutes on a Pi 5)..."
  GENERATOR=$(command -v ninja >/dev/null && echo Ninja || echo "Unix Makefiles")
  # The server only: optimized for this CPU, no web UI, nothing fetched at build or run time.
  cmake -S "$SOURCE" -B "$BUILD" -G "$GENERATOR" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF \
    > "$DATA/llama-build.log" 2>&1 || die "configuring llama.cpp failed: see $DATA/llama-build.log"
  cmake --build "$BUILD" --config Release --target llama-server -j "$(nproc 2>/dev/null || echo 4)" \
    >> "$DATA/llama-build.log" 2>&1 || die "building llama-server failed: see $DATA/llama-build.log"
  echo "$STAMP" > "$BUILD/STAMP"
fi
echo "llama-server: $SERVER_BIN ($TAG)"

# 3. The model file, checked against DigUp's checksum.
MODEL="$MODELS/$FILE"
if [ -f "$MODEL" ] && [ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$SHA256" ]; then
  :
else
  echo "Downloading $FILE (310 MB, resumable)..."
  curl -fL --retry 5 --retry-delay 3 -C - -o "$MODEL.part" "https://huggingface.co/$REPO_ID/resolve/$REVISION/$FILE"
  if [ "$(sha256sum "$MODEL.part" | cut -d' ' -f1)" != "$SHA256" ]; then
    rm -f "$MODEL.part"
    die "the download's checksum doesn't match DigUp's; deleted it. Run this again."
  fi
  mv "$MODEL.part" "$MODEL"
fi
[ "$(stat -c %s "$MODEL")" = "$BYTES" ] || die "$MODEL isn't the size DigUp's file is"
echo "model: $MODEL"

# 4. The service. 127.0.0.1 only. -c/-b/-ub 2048: one input is processed whole (EmbeddingGemma is non-causal), as
# in DigUp, so documents as long as DigUp's passages fit too (doctor compares those).
mkdir -p "$(dirname "$UNIT_FILE")"
cat > "$UNIT_FILE" <<EOF
[Unit]
Description=EmbeddingGemma 2 for hermes-digup (llama.cpp $TAG)
After=network.target

[Service]
ExecStart="$SERVER_BIN" --model "$MODEL" --embedding --pooling mean --ctx-size 2048 --batch-size 2048 --ubatch-size 2048 --parallel 1 --threads $THREADS --threads-batch $THREADS --host 127.0.0.1 --port $PORT --no-webui
Restart=always
RestartSec=5
Nice=5
MemoryMax=1500M

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable "$SERVICE" >/dev/null 2>&1
systemctl --user restart "$SERVICE"
# A user service runs at boot only with lingering on; otherwise it starts at login and stops at logout.
if [ "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null)" != yes ]; then
  echo "Letting the service run without a login session (sudo loginctl enable-linger)..."
  sudo loginctl enable-linger "$(id -un)" || echo "warning: run 'sudo loginctl enable-linger $(id -un)' so it starts at boot" >&2
fi

echo -n "Starting"
ready=0
for _ in $(seq 1 120); do
  if [ "$(curl -s -o /dev/null -m 2 -w '%{http_code}' "$LOCAL_URL/health" || true)" = 200 ]; then ready=1; break; fi
  echo -n "."
  sleep 1
done
echo
if [ "$ready" != 1 ]; then
  echo "The model server didn't start. Its log:" >&2
  journalctl --user -u "$SERVICE" -n 30 --no-pager >&2 || true
  exit 1
fi

# A test query, made the way hermes-digup makes it.
RESPONSE="$(curl -sS -m 60 -H 'Content-Type: application/json' \
  -d '{"input": ["task: search result | query: a dog on the beach"], "model": "embeddinggemma-2"}' \
  -w '\n%{time_total}' "$LOCAL_URL/v1/embeddings" || true)"
SECONDS_TAKEN="${RESPONSE##*$'\n'}"
DIMS="$(printf '%s' "${RESPONSE%$'\n'*}" | python3 -c '
import json, sys
try:
    print(len(json.load(sys.stdin)["data"][0]["embedding"]))
except Exception:
    print(0)')"
[ "$DIMS" = 768 ] || die "the test query didn't come back as a 768-dimension vector (got $DIMS)"
echo "EmbeddingGemma 2 is answering on this Pi: 768 dimensions, ${SECONDS_TAKEN}s for a test query."

# 5. This Pi first; the Mac's addresses after it, used only if this server is down.
if [ "$FALLBACK" = 1 ]; then
  set_urls "$LOCAL_URL,$(other_urls)"
else
  set_urls "$LOCAL_URL"
fi

if command -v hermes-digup >/dev/null; then
  echo
  hermes-digup doctor 2>/dev/null || true
fi
cat <<EOF

Next:
  - In Hermes, run /reload-mcp (or restart Hermes): the MCP server reads DIGUP_EMBED_URL when it starts.
  - To free the Mac's memory, unpin EmbeddingGemma 2 there: deploy/mac-server/setup-embedding-server.sh --unpin
    It keeps the Pi's key, so the Mac still works as the fallback (oMLX loads the model again only if it's used).
Service: systemctl --user status $SERVICE    Log: journalctl --user -u $SERVICE
EOF
