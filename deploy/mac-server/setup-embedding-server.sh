#!/bin/bash
# Runs EmbeddingGemma 2 on this Mac (Mac no.2, the always-on one) as an embedding server for the Pi.
#
# It serves the exact file DigUp uses (ggml-org's embeddinggemma-2-Q8_0.gguf at DigUp's pinned
# revision) with llama.cpp's llama-server, mean-pooled like DigUp, so the Pi's query vectors line up
# with the vectors in DigUp's index. A launchd job keeps it running and restarts it if it stops.
#
#   deploy/mac-server/setup-embedding-server.sh [--port 8088] [--host 0.0.0.0] [--no-api-key]
#   deploy/mac-server/setup-embedding-server.sh --uninstall

set -euo pipefail

LABEL="com.hermes-digup.embeddings"
PORT=8088
HOST=0.0.0.0
USE_KEY=1
UNINSTALL=0
SUPPORT="$HOME/Library/Application Support/hermes-digup"
CONFIG_DIR="$HOME/.config/hermes-digup"
KEY_FILE="$CONFIG_DIR/embedding-server.env"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/hermes-digup-embeddings.log"

# DigUp's model (Sources/DigUpKit/ModelFiles.swift).
REPO="ggml-org/embeddinggemma-2-GGUF"
REVISION="bfcd298762cc34d0357ece5ebdd31791a3a374d8"
FILE="embeddinggemma-2-Q8_0.gguf"
SHA256="2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135"
BYTES=309855456
MIN_BUILD=11452   # the first llama.cpp builds with EmbeddingGemma 2

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --no-api-key) USE_KEY=0; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ "$(uname -s)" = Darwin ] || { echo "This script is for macOS." >&2; exit 1; }
if [ "$UNINSTALL" = 1 ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Stopped and removed the embedding server job (the model stays in $SUPPORT/models)."
  exit 0
fi
mkdir -p "$SUPPORT/models" "$CONFIG_DIR" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"

# 1. llama-server
SERVER="$(command -v llama-server || true)"
for candidate in /opt/homebrew/bin/llama-server /usr/local/bin/llama-server; do
  [ -z "$SERVER" ] && [ -x "$candidate" ] && SERVER="$candidate"
done
if [ -z "$SERVER" ]; then
  BREW="$(command -v brew || echo /opt/homebrew/bin/brew)"
  [ -x "$BREW" ] || { echo "Install Homebrew (https://brew.sh) or llama.cpp first." >&2; exit 1; }
  echo "Installing llama.cpp with Homebrew..."
  "$BREW" install llama.cpp
  SERVER="$("$BREW" --prefix)/bin/llama-server"
fi
BUILD="$("$SERVER" --version 2>&1 | sed -n 's/^version: \([0-9][0-9]*\).*/\1/p' | head -1)"
if [ -n "$BUILD" ] && [ "$BUILD" -lt "$MIN_BUILD" ]; then
  echo "llama.cpp build $BUILD is too old for EmbeddingGemma 2 (needs $MIN_BUILD+). Run: brew upgrade llama.cpp" >&2
  exit 1
fi
echo "llama-server: $SERVER (build ${BUILD:-unknown})"

# 2. The model: reuse DigUp's or the Hugging Face cache's copy if this Mac has one, else download it.
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
MODEL=""
for candidate in \
  "$HOME/Library/Application Support/DigUp/Models/$FILE" \
  "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ggml-org--embeddinggemma-2-GGUF/snapshots/$REVISION/$FILE" \
  "$SUPPORT/models/$FILE"; do
  if [ -f "$candidate" ] && [ "$(stat -L -f %z "$candidate")" = "$BYTES" ]; then
    MODEL="$candidate"
    break
  fi
done
if [ -z "$MODEL" ]; then
  MODEL="$SUPPORT/models/$FILE"
  echo "Downloading $FILE (310 MB, resumable)..."
  curl -fL --retry 5 --retry-delay 3 -C - -o "$MODEL.part" \
    "https://huggingface.co/$REPO/resolve/$REVISION/$FILE"
  if [ "$(sha_of "$MODEL.part")" != "$SHA256" ]; then
    rm -f "$MODEL.part"
    echo "The download's checksum doesn't match; deleted it. Run this again." >&2
    exit 1
  fi
  mv "$MODEL.part" "$MODEL"
elif [ "$(sha_of "$MODEL")" != "$SHA256" ]; then
  echo "$MODEL doesn't match DigUp's checksum; not using it." >&2
  exit 1
fi
echo "model: $MODEL"

# 3. An API key, so only the Pi (who has the key) can use it.
KEY=""
if [ "$USE_KEY" = 1 ]; then
  if [ -f "$KEY_FILE" ]; then
    KEY="$(sed -n 's/^DIGUP_EMBED_API_KEY=//p' "$KEY_FILE")"
  fi
  if [ -z "$KEY" ]; then
    KEY="$(openssl rand -hex 24)"
    umask 077
    echo "DIGUP_EMBED_API_KEY=$KEY" > "$KEY_FILE"
  fi
fi

# 4. The launchd job. -c/-b/-ub 2048: one input is processed whole (EmbeddingGemma is non-causal), as in DigUp.
{
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$SERVER</string>
    <string>--model</string><string>$MODEL</string>
    <string>--embedding</string>
    <string>--pooling</string><string>mean</string>
    <string>--ctx-size</string><string>2048</string>
    <string>--batch-size</string><string>2048</string>
    <string>--ubatch-size</string><string>2048</string>
    <string>--parallel</string><string>1</string>
    <string>--n-gpu-layers</string><string>999</string>
    <string>--host</string><string>$HOST</string>
    <string>--port</string><string>$PORT</string>
EOF
  [ -n "$KEY" ] && echo "    <string>--api-key</string><string>$KEY</string>"
  cat <<EOF
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
} > "$PLIST"
chmod 600 "$PLIST"   # it holds the API key
plutil -lint "$PLIST" >/dev/null
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

# 5. Wait for it and make a test vector.
echo -n "Starting"
for _ in $(seq 1 60); do
  if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  echo -n "."
  sleep 1
done
echo
AUTH=()
[ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")
RESPONSE="$(curl -fs ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' \
  -d '{"input": ["task: search result | query: a dog on the beach"], "model": "embeddinggemma-2"}' \
  "http://127.0.0.1:$PORT/v1/embeddings" || true)"
DIMS="$(printf '%s' "$RESPONSE" | sed 's/.*"embedding":\[\([^]]*\)\].*/\1/' | tr ',' '\n' | grep -c . || true)"
if [ "$DIMS" != 768 ]; then
  echo "The server didn't return a 768-dimension vector. Last lines of $LOG:" >&2
  tail -20 "$LOG" >&2
  exit 1
fi
echo "Embedding server is up: 768-dimension vectors on port $PORT."

# How the Pi reaches this Mac: its MagicDNS name, then its Tailscale IP as a fallback.
TS="$(command -v tailscale || true)"
[ -z "$TS" ] && [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ] && TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
URLS=""
if [ -n "$TS" ]; then
  TS_NAME="$("$TS" status --peers=false 2>/dev/null | awk 'NR==1 {print $2}')"
  TS_IP="$("$TS" ip -4 2>/dev/null | head -1)"
  [ -n "$TS_NAME" ] && URLS="http://$TS_NAME:$PORT"
  [ -n "$TS_IP" ] && URLS="${URLS:+$URLS,}http://$TS_IP:$PORT"
fi
[ -n "$URLS" ] || URLS="http://$(scutil --get LocalHostName 2>/dev/null || hostname):$PORT"
cat <<EOF

Put these in the Pi's ~/.config/hermes-digup/config.env:

  DIGUP_EMBED_URL=$URLS
EOF
[ -n "$KEY" ] && echo "  DIGUP_EMBED_API_KEY=$KEY"
cat <<EOF

Keep this Mac awake (System Settings > Energy: prevent automatic sleeping; or: sudo pmset -a sleep 0)
and logged in (launchd agents run while you're logged in: turn on automatic login for a headless Mac).
EOF
