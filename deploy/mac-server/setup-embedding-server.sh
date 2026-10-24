#!/bin/bash
# Sets up oMLX on this Mac (Mac no.2, the always-on one) to serve EmbeddingGemma 2 to the Pi.
#
# oMLX loads google/embeddinggemma-2 through mlx-vlm, the reference DigUp measured its own vectors against
# (cosine >= 0.999), so the Pi's query vectors line up with the vectors in DigUp's index. Everything goes
# through oMLX's admin API, so your other models and settings stay as they are:
#
#   1. finds your oMLX server (the app or `brew services`) and starts it if it isn't running;
#   2. downloads EmbeddingGemma 2 into oMLX's model folder, unless oMLX already has it;
#   3. pins it, so it stays loaded and the first search after a quiet spell is quick;
#   4. creates an oMLX sub key named "hermes-digup" for the Pi: it can run models, not change settings;
#   5. checks that a test query comes back as a 768-dimension vector, and prints the Pi's settings.
#
#   deploy/mac-server/setup-embedding-server.sh [--expose] [--admin-key KEY] [--port 8000] [--hf-token TOKEN]
#   deploy/mac-server/setup-embedding-server.sh --uninstall      # delete the Pi's sub key and unpin the model
#
# --expose   oMLX listens only on this Mac by default (127.0.0.1). This sets its host to 0.0.0.0 so the Pi can
#            reach it over Tailscale (oMLX then refuses requests without a key) and restarts oMLX.

set -euo pipefail

REPO_ID="google/embeddinggemma-2"
KEY_NAME="hermes-digup"
CONFIG_DIR="$HOME/.config/hermes-digup"
KEY_FILE="$CONFIG_DIR/embedding-server.env"
EXPOSE=0
UNINSTALL=0
PORT=""
ADMIN_KEY="${OMLX_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"
MODEL_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --expose) EXPOSE=1; shift ;;
    --admin-key) ADMIN_KEY="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ "$(uname -s)" = Darwin ] || { echo "This script is for macOS." >&2; exit 1; }
die() { echo "error: $*" >&2; exit 1; }

# JSON without jq or Python: macOS's own JavaScript. `json_get JSON a.b.*.c` prints each match on a line
# (objects as JSON); `*` walks every item of an array or object.
JSON_JS='function run(argv) {
  var values = [JSON.parse(argv[0])];
  (argv[1] ? argv[1].split(".") : []).forEach(function (part) {
    var next = [];
    values.forEach(function (v) {
      if (v === null || typeof v !== "object") return;
      if (part === "*") {
        (Array.isArray(v) ? v : Object.keys(v).map(function (k) { return v[k]; }))
          .forEach(function (x) { next.push(x); });
      } else if (v[part] !== undefined) {
        next.push(v[part]);
      }
    });
    values = next;
  });
  return values.filter(function (v) { return v !== null && v !== undefined; })
    .map(function (v) { return typeof v === "object" ? JSON.stringify(v) : String(v); }).join("\n");
}'
json_get() { osascript -l JavaScript -e "$JSON_JS" "$1" "$2" 2>/dev/null; }

# 1. oMLX's settings: where it listens, and its main key (needed for the admin API).
BASE="${OMLX_BASE_PATH:-}"
BOOTSTRAP="$HOME/Library/Application Support/oMLX/base-path"
[ -z "$BASE" ] && [ -s "$BOOTSTRAP" ] && BASE="$(head -1 "$BOOTSTRAP")"
BASE="${BASE:-$HOME/.omlx}"
BASE="${BASE/#\~/$HOME}"
SETTINGS_JSON="{}"
[ -f "$BASE/settings.json" ] && SETTINGS_JSON="$(cat "$BASE/settings.json")"
setting() { json_get "$SETTINGS_JSON" "$1" | head -1; }
PORT="${PORT:-${OMLX_PORT:-$(setting server.port)}}"
PORT="${PORT:-8000}"
BIND="$(setting server.host)"
BIND="${BIND:-127.0.0.1}"
ADMIN_KEY="${ADMIN_KEY:-$(setting auth.api_key)}"

# The host may be a comma-separated list. Reach oMLX locally through loopback when it listens there.
LOCAL_HOST=""
EXPOSED=0
IFS=',' read -r -a BIND_PARTS <<< "$BIND"
for part in "${BIND_PARTS[@]}"; do
  part="$(echo "$part" | tr -d ' ')"
  case "$part" in
    127.0.0.1|localhost|::1) LOCAL_HOST="127.0.0.1" ;;
    0.0.0.0|::) LOCAL_HOST="127.0.0.1"; EXPOSED=1 ;;
    "") ;;
    *) EXPOSED=1; [ -z "$LOCAL_HOST" ] && LOCAL_HOST="$part" ;;
  esac
done
LOCAL_HOST="${LOCAL_HOST:-127.0.0.1}"
OMLX="http://$LOCAL_HOST:$PORT"

health() { curl -s -o /dev/null -m 5 -w '%{http_code}' "$OMLX/health" || true; }
wait_ready() {
  local code
  for _ in $(seq 1 "${1:-90}"); do
    code="$(health)"
    [ "$code" = 200 ] && return 0
    sleep 1
  done
  return 1
}

# 2. A running server. The app or `brew services` restarts it if it ever stops.
if [ "$(health)" != 200 ]; then
  OMLX_CLI="$(command -v omlx || true)"
  for candidate in "$HOME/.omlx/bin/omlx" /opt/homebrew/bin/omlx /usr/local/bin/omlx; do
    [ -z "$OMLX_CLI" ] && [ -x "$candidate" ] && OMLX_CLI="$candidate"
  done
  if [ "$(health)" = 503 ]; then
    echo "oMLX is loading its pinned models..."
  elif [ -n "$OMLX_CLI" ]; then
    echo "Starting oMLX..."
    "$OMLX_CLI" start >/dev/null 2>&1 || true
  elif [ -d /Applications/oMLX.app ]; then
    echo "Opening oMLX..."
    open -a oMLX
  else
    die "oMLX isn't installed or running on port $PORT. Install it (https://github.com/jundot/omlx) and run this again."
  fi
  wait_ready 120 || die "oMLX didn't answer on $OMLX (see ~/.omlx/logs/server.log). Pass --port if it uses another port."
fi
echo "oMLX: $OMLX"

[ -n "$ADMIN_KEY" ] || die "oMLX has no main API key yet, and the Pi needs one to reach it over the network.
Set one in oMLX (menu bar app: Settings > Security, or the web admin at $OMLX/admin), then run this again.
If you set it with OMLX_API_KEY instead of in settings, pass --admin-key KEY."

BODY=""
# api METHOD PATH [JSON]: calls the admin API with the main key. The response is in $BODY.
api() {
  local args=(-sS -m 60 -X "$1" -H "Authorization: Bearer $ADMIN_KEY" -w $'\n%{http_code}')
  [ -n "${3:-}" ] && args+=(-H "Content-Type: application/json" --data "$3")
  local out code
  out="$(curl "${args[@]}" "$OMLX$2")" || { BODY="(no answer)"; return 1; }
  code="${out##*$'\n'}"
  BODY="${out%$'\n'*}"
  [ "$code" -ge 200 ] && [ "$code" -lt 300 ]
}
api GET /admin/api/server-info || die "oMLX refused the admin key: $BODY
Pass the main API key with --admin-key (sub keys can't use the admin API)."

# --uninstall: take the Pi's key away and unpin the model (the model itself stays, delete it in oMLX).
if [ "$UNINSTALL" = 1 ]; then
  if [ -f "$KEY_FILE" ]; then
    key="$(sed -n 's/^DIGUP_EMBED_API_KEY=//p' "$KEY_FILE")"
    [ -n "$key" ] && api DELETE /admin/api/sub-keys "{\"key\": \"$key\"}" && echo "Deleted the Pi's sub key."
    model="$(sed -n 's/^DIGUP_EMBED_MODEL=//p' "$KEY_FILE")"
    [ -n "$model" ] && api PUT "/admin/api/models/$model/settings" '{"is_pinned": false}' && echo "Unpinned $model."
    rm -f "$KEY_FILE"
  fi
  echo "Done. The Pi can no longer use this Mac's oMLX."
  exit 0
fi

# Network: the Pi must reach oMLX over Tailscale.
if [ "$EXPOSED" = 0 ]; then
  if [ "$EXPOSE" = 1 ]; then
    echo "Making oMLX listen on the network (0.0.0.0, key required) and restarting it..."
    api POST /admin/api/global-settings '{"host": "0.0.0.0"}' || die "couldn't change oMLX's host: $BODY"
    if api POST /admin/api/server/restart; then
      sleep 3
      wait_ready 180 || die "oMLX didn't come back after the restart; check it in the menu bar."
    else
      echo "oMLX can't restart itself here (it isn't run by the app or brew services). Restart it now, then"
      echo "run this script again."
      exit 1
    fi
  else
    die "oMLX only listens on this Mac ($BIND), so the Pi can't reach it. Run this again with --expose,
or set the server host to 0.0.0.0 in oMLX's settings and restart it."
  fi
fi

# 3. EmbeddingGemma 2 in oMLX's model folder. find_model reads the model list in $BODY.
find_model() {
  local model id
  while IFS= read -r model; do
    [ -n "$model" ] || continue
    [ "$(json_get "$model" model_type)" = embedding ] || continue
    id="$(json_get "$model" id)"
    if printf '%s %s' "$id" "$(json_get "$model" source_repo_id)" | grep -qi 'embeddinggemma-\?2'; then
      echo "$id"
      return 0
    fi
  done <<< "$(json_get "$BODY" 'models.*')"
  return 1
}
if [ -z "$MODEL_ID" ]; then
  api GET /admin/api/models || die "couldn't list oMLX's models: $BODY"
  MODEL_ID="$(find_model || true)"
fi
if [ -z "$MODEL_ID" ]; then
  echo "Downloading $REPO_ID with oMLX (about 1.5 GB)..."
  api POST /admin/api/hf/download "{\"repo_id\": \"$REPO_ID\", \"hf_token\": \"$HF_TOKEN\"}" \
    || die "oMLX couldn't start the download: $BODY"
  last=""
  while true; do
    sleep "${OMLX_POLL_SECONDS:-3}"
    api GET /admin/api/hf/tasks || die "couldn't read oMLX's download progress: $BODY"
    status="" progress="" error=""
    while IFS= read -r task; do
      [ "$(json_get "$task" repo_id)" = "$REPO_ID" ] || continue
      status="$(json_get "$task" status)"
      progress="$(json_get "$task" progress)"
      error="$(json_get "$task" error)"
    done <<< "$(json_get "$BODY" 'tasks.*')"
    case "$status" in
      completed) echo; break ;;
      failed|cancelled)
        echo
        die "the download $status: ${error:-see the oMLX server log}
If Hugging Face says the model is gated, accept its license at https://huggingface.co/$REPO_ID and pass
--hf-token with a token from https://huggingface.co/settings/tokens." ;;
      *) [ "$progress" != "$last" ] && printf '\r  %s%%' "${progress:-0}" && last="$progress" ;;
    esac
  done
  api POST /admin/api/reload || true
  api GET /admin/api/models || die "couldn't list oMLX's models: $BODY"
  MODEL_ID="$(find_model || true)"
  [ -n "$MODEL_ID" ] || die "the download finished but oMLX doesn't list an EmbeddingGemma 2 model. Check
oMLX's Models page; if it's there under another name, run this again with --model-id NAME."
fi
echo "model: $MODEL_ID"

# Pinned: kept in memory (text only, about 0.9 GB), never unloaded for being idle.
api PUT "/admin/api/models/$MODEL_ID/settings" '{"is_pinned": true}' || echo "warning: couldn't pin $MODEL_ID: $BODY"
api POST "/admin/api/models/$MODEL_ID/load" || echo "warning: couldn't load $MODEL_ID yet: $BODY"

# 4. The Pi's own key: a sub key runs models but can't touch oMLX's settings, and can be revoked alone.
mkdir -p "$CONFIG_DIR"
KEY=""
[ -f "$KEY_FILE" ] && KEY="$(sed -n 's/^DIGUP_EMBED_API_KEY=//p' "$KEY_FILE")"
key_works() {
  [ -n "$1" ] && [ "$(curl -s -o /dev/null -m 10 -w '%{http_code}' -H "Authorization: Bearer $1" "$OMLX/v1/models")" = 200 ]
}
if ! key_works "$KEY"; then
  KEY="hd-$(openssl rand -hex 24)"
  api POST /admin/api/sub-keys "{\"key\": \"$KEY\", \"name\": \"$KEY_NAME\"}" || die "couldn't create a sub key: $BODY"
fi
umask 077
printf 'DIGUP_EMBED_API_KEY=%s\nDIGUP_EMBED_MODEL=%s\n' "$KEY" "$MODEL_ID" > "$KEY_FILE"

# 5. A test query, made the way the Pi will make it.
RESPONSE="$(curl -sS -m 120 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"input\": [\"task: search result | query: a dog on the beach\"], \"model\": \"$MODEL_ID\"}" \
  "$OMLX/v1/embeddings" || true)"
DIMS="$(json_get "$RESPONSE" 'data.0.embedding.*' | grep -c . || true)"
if [ "$DIMS" != 768 ]; then
  echo "The test query didn't come back as a 768-dimension vector (got ${DIMS:-0}):" >&2
  printf '%s\n' "$RESPONSE" | head -c 600 >&2
  echo >&2
  exit 1
fi
echo "EmbeddingGemma 2 is answering: 768-dimension vectors."

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

Put these in the Pi's ~/.config/hermes-digup/config.env (deploy/pi/install.sh takes the first two):

  DIGUP_EMBED_URL=$URLS
  DIGUP_EMBED_API_KEY=$KEY
  DIGUP_EMBED_MODEL=$MODEL_ID

They're also saved in $KEY_FILE.
Keep this Mac awake (System Settings > Energy: prevent automatic sleeping; or: sudo pmset -a sleep 0),
logged in (automatic login for a headless Mac), and oMLX starting at login (its menu bar app does by default;
with Homebrew: brew services start omlx).
EOF
