#!/bin/bash
# Installs hermes-digup on the Raspberry Pi that runs Hermes Agent.
#
#   deploy/pi/install.sh [--embed-url http://mac-mini:8088] [--api-key KEY]
#
# Puts the server in its own virtualenv (~/.local/share/hermes-digup/venv), links the command to
# ~/.local/bin/hermes-digup, creates the folder the Mac syncs the index into, writes
# ~/.config/hermes-digup/config.env, and prints the lines to add to Hermes.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="$HOME/.local/share/hermes-digup"
VENV="$DATA/venv"
INDEX_DIR="$DATA/index"
CONFIG_DIR="$HOME/.config/hermes-digup"
CONFIG="$CONFIG_DIR/config.env"
BIN="$HOME/.local/bin"
EMBED_URL=""
API_KEY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --embed-url) EMBED_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

PYTHON="$(command -v python3 || true)"
[ -n "$PYTHON" ] || { echo "python3 is missing: sudo apt install python3" >&2; exit 1; }
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
  echo "Python 3.10+ is needed (Raspberry Pi OS Bookworm or newer has it)." >&2
  exit 1
fi
missing=()
"$PYTHON" -c 'import venv, ensurepip' 2>/dev/null || missing+=(python3-venv)
command -v rsync >/dev/null || missing+=(rsync)
if [ ${#missing[@]} -gt 0 ]; then
  echo "Installing ${missing[*]} (needs sudo)..."
  sudo apt-get update -qq && sudo apt-get install -y "${missing[@]}"
fi
# DigUp's keyword index needs SQLite's FTS5 with the unicode61 'categories' option (SQLite 3.25+).
"$PYTHON" - <<'EOF'
import sqlite3, sys
db = sqlite3.connect(":memory:")
try:
    db.execute("""CREATE VIRTUAL TABLE t USING fts5(a, tokenize = "porter unicode61 remove_diacritics 2 categories 'L* N* Co M*'")""")
except sqlite3.Error as error:
    sys.exit(f"This Python's SQLite ({sqlite3.sqlite_version}) can't read DigUp's keyword index: {error}")
EOF

echo "Installing into $VENV..."
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet --upgrade "$REPO"
mkdir -p "$BIN" "$INDEX_DIR" "$CONFIG_DIR"
ln -sf "$VENV/bin/hermes-digup" "$BIN/hermes-digup"

if [ ! -f "$CONFIG" ]; then
  umask 077
  cat > "$CONFIG" <<EOF
# hermes-digup settings (read by the MCP server and by \`hermes-digup doctor\`).
# Where the Mac's sync puts DigUp's index (PI_INDEX_DIR in the Mac's sync.env).
DIGUP_INDEX_DIR=$INDEX_DIR
# The always-on Mac's embedding server; several, comma-separated, are tried in order.
DIGUP_EMBED_URL=$EMBED_URL
# The key setup-embedding-server.sh printed (empty if you used --no-api-key).
DIGUP_EMBED_API_KEY=$API_KEY
# openai = llama-server / LM Studio / Ollama's /v1 API; ollama = Ollama's /api/embed.
DIGUP_EMBED_API=openai
DIGUP_EMBED_MODEL=embeddinggemma-2
DIGUP_EMBED_TIMEOUT=15
# Searches mention it when the index is older than this (the laptop was away or asleep).
DIGUP_STALE_HOURS=36
EOF
  echo "Wrote $CONFIG"
else
  echo "Keeping existing $CONFIG"
  [ -n "$EMBED_URL" ] && sed -i "s|^DIGUP_EMBED_URL=.*|DIGUP_EMBED_URL=$EMBED_URL|" "$CONFIG"
  [ -n "$API_KEY" ] && sed -i "s|^DIGUP_EMBED_API_KEY=.*|DIGUP_EMBED_API_KEY=$API_KEY|" "$CONFIG"
fi

cat <<EOF

Installed: $BIN/hermes-digup ($("$BIN/hermes-digup" --version))

Add it to Hermes, either with
  hermes mcp add digup --command $BIN/hermes-digup --args serve
or in ~/.hermes/config.yaml:

  mcp_servers:
    digup:
      command: "$BIN/hermes-digup"
      args: ["serve"]
      timeout: 60
      connect_timeout: 30

then /reload-mcp in Hermes (a running gateway picks the change up by itself within a minute).
Check everything with:  hermes-digup doctor
EOF
