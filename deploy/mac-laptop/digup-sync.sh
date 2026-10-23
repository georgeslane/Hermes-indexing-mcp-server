#!/bin/bash
# Copies DigUp's index from this Mac to the Raspberry Pi, safely, while DigUp keeps running.
#
#   1. Takes a consistent snapshot with SQLite's backup API (DigUp may be writing; its WAL is honoured).
#   2. Checks the snapshot (quick_check, schema) before anything leaves this Mac.
#   3. rsyncs it to the Pi over SSH. rsync writes a temp file and renames it into place, so the
#      server on the Pi only ever sees a whole file; only changed blocks cross the network.
#   4. Writes manifest.json (when, from where) so the Pi can tell how fresh its copy is.
#
# Run by launchd every few minutes (see install.sh). Safe to run by hand: `digup-sync.sh --force -v`.
# Written for macOS's own /bin/bash 3.2, sqlite3, rsync and ssh: nothing to install.

set -u -o pipefail

CONFIG="${HERMES_DIGUP_SYNC_CONFIG:-$HOME/.config/hermes-digup/sync.env}"
STATE_DIR="$HOME/Library/Application Support/hermes-digup"
LOG_FILE="$HOME/Library/Logs/hermes-digup-sync.log"
FORCE=0
VERBOSE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -v|--verbose) VERBOSE=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# shellcheck source=/dev/null
[ -f "$CONFIG" ] && . "$CONFIG"
PI_HOST="${PI_HOST:-}"                                        # user@host, e.g. pi@raspberrypi (MagicDNS name)
PI_INDEX_DIR="${PI_INDEX_DIR:-.local/share/hermes-digup/index}"  # on the Pi; relative to its home
PI_INDEX_DIR="${PI_INDEX_DIR#\~/}"
SSH_KEY="${SSH_KEY:-}"
SKIP_ON_BATTERY="${SKIP_ON_BATTERY:-0}"
SYNC_CODE_INDEX="${SYNC_CODE_INDEX:-1}"
SQLITE="${SQLITE:-/usr/bin/sqlite3}"

mkdir -p "$STATE_DIR" "$(dirname "$LOG_FILE")"
if [ -f "$LOG_FILE" ] && [ "$(stat -f %z "$LOG_FILE")" -gt 1048576 ]; then
  mv -f "$LOG_FILE" "$LOG_FILE.1"
fi

log() {
  local line
  line="$(date '+%Y-%m-%d %H:%M:%S') $*"
  echo "$line" >> "$LOG_FILE"
  # stderr: stdout carries manifest entries between functions.
  if [ "$VERBOSE" = 1 ] || [ -t 2 ]; then echo "$line" >&2; fi
}
die() { log "ERROR: $*"; [ "$VERBOSE" = 1 ] || [ -t 2 ] || echo "ERROR: $*" >&2; exit "${2:-1}"; }

[ -n "$PI_HOST" ] || die "PI_HOST isn't set. Put PI_HOST=user@pi-hostname in $CONFIG (install.sh does this)."
[ -x "$SQLITE" ] || die "no sqlite3 at $SQLITE"

# One run at a time: launchd can start a run while a slow one is still uploading.
LOCK="$STATE_DIR/sync.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -f "$LOCK/pid" ] && kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
    log "another sync is running (pid $(cat "$LOCK/pid")); skipping"
    exit 0
  fi
  rm -rf "$LOCK"   # left behind by a run that was killed
  mkdir "$LOCK" || die "can't take the lock $LOCK"
fi
echo $$ > "$LOCK/pid"
STAGING="$STATE_DIR/staging"
# shellcheck disable=SC2329  # run by the EXIT trap
cleanup() { rm -rf "$LOCK" "$STAGING"; }
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ "$SKIP_ON_BATTERY" = 1 ] && pmset -g batt 2>/dev/null | grep -q "Battery Power"; then
  log "on battery; skipping (SKIP_ON_BATTERY=1)"
  exit 0
fi

# Where DigUp keeps its index: DIGUP_INDEX_DIR, else DigUp's own setting, else its default.
find_index_dir() {
  if [ -n "${DIGUP_INDEX_DIR:-}" ]; then
    echo "${DIGUP_INDEX_DIR/#\~/$HOME}"
    return
  fi
  local custom
  custom="$(defaults read com.abdurrahim.DigUp indexDir 2>/dev/null || true)"
  if [ -n "$custom" ]; then
    custom="${custom#file://}"
    custom="$(printf '%s' "$custom" | sed 's/%20/ /g')"
    echo "${custom/#\~/$HOME}"
    return
  fi
  echo "$HOME/Library/Application Support/DigUp/index.noindex"
}
INDEX_DIR="$(find_index_dir)"
[ -f "$INDEX_DIR/index.sqlite" ] || die "no DigUp index at $INDEX_DIR/index.sqlite. Has DigUp finished setup? \
If you moved its index, set DIGUP_INDEX_DIR in $CONFIG."

SSH_CMD=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
[ -n "$SSH_KEY" ] && SSH_CMD+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
remote() { "${SSH_CMD[@]}" "$PI_HOST" "$@"; }
REMOTE_DIR="$(printf '%q' "$PI_INDEX_DIR")"

if ! remote "mkdir -p $REMOTE_DIR" 2>>"$LOG_FILE"; then
  log "the Pi ($PI_HOST) isn't reachable right now; will try again next run"
  exit 75
fi

# The source's size and dates: unchanged since the last good upload means nothing to send.
stamp_of() {
  stat -f '%m:%z' "$1" "$1-wal" 2>/dev/null | tr '\n' ' '
}

json_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# snapshot SOURCE DEST: a consistent, self-contained, checked copy. Prints "schema files vectors".
snapshot() {
  local src="$1" dst="$2" check
  rm -f "$dst" "$dst-journal" "$dst-wal" "$dst-shm"
  if ! "$SQLITE" -bail -readonly "$src" >/dev/null 2>>"$LOG_FILE" <<EOF
.timeout 60000
.backup '$dst'
EOF
  then
    return 1
  fi
  # A file of its own (no -wal beside it), so the Pi can open it read-only and immutable.
  "$SQLITE" "$dst" "PRAGMA journal_mode=DELETE;" >/dev/null 2>>"$LOG_FILE" || return 1
  check="$("$SQLITE" "$dst" "PRAGMA quick_check;" 2>>"$LOG_FILE")"
  if [ "$check" != "ok" ]; then
    log "snapshot of $src failed its integrity check: $(echo "$check" | head -3)"
    return 1
  fi
  "$SQLITE" -separator ' ' "$dst" \
    "SELECT (SELECT user_version FROM pragma_user_version), (SELECT COUNT(*) FROM files),
            (SELECT COUNT(*) FROM segments WHERE vec IS NOT NULL);" 2>>"$LOG_FILE"
}

# sync_one NAME: index.sqlite or code.sqlite. Prints a manifest entry on success.
sync_one() {
  local name="$1" src="$INDEX_DIR/$1" stamp_file="$STATE_DIR/last-stamp-$1" stamp counts started
  stamp="$(stamp_of "$src")"
  if [ "$FORCE" = 0 ] && [ -f "$stamp_file" ] && [ "$(cat "$stamp_file")" = "$stamp" ] \
     && [ -f "$STATE_DIR/last-entry-$name" ]; then
    [ "$VERBOSE" = 1 ] && log "$name unchanged since the last sync"
    cat "$STATE_DIR/last-entry-$name"
    return 0
  fi
  mkdir -p "$STAGING"
  started=$(date +%s)
  if ! counts="$(snapshot "$src" "$STAGING/$name")"; then
    log "couldn't snapshot $src; the Pi keeps its previous copy"
    return 1
  fi
  # shellcheck disable=SC2086  # three numbers, split on purpose
  set -- $counts
  local schema="$1" files="$2" vectors="$3" bytes mtime
  bytes="$(stat -f %z "$STAGING/$name")"
  # rsync skips a file whose size and date match the Pi's copy. Two snapshots made in the same second can
  # have the same size, so make sure this one's date differs from the last upload's.
  mtime="$(stat -f %m "$STAGING/$name")"
  if [ "$mtime" = "$(cat "$STATE_DIR/last-mtime-$name" 2>/dev/null)" ]; then
    sleep 1
    touch "$STAGING/$name"
    mtime="$(stat -f %m "$STAGING/$name")"
  fi
  if ! rsync -t -e "${SSH_CMD[*]}" "$STAGING/$name" "$PI_HOST:$PI_INDEX_DIR/$name" 2>>"$LOG_FILE"; then
    log "upload of $name to $PI_HOST failed; the Pi keeps its previous copy"
    return 1
  fi
  rm -f "$STAGING/$name"
  echo "$mtime" > "$STATE_DIR/last-mtime-$name"
  log "synced $name: $files files, $vectors vectors, $((bytes / 1048576)) MB in $(( $(date +%s) - started )) s"
  printf '"%s": {"bytes": %s, "schema": %s, "files": %s, "vectors": %s}' \
    "$name" "$bytes" "$schema" "$files" "$vectors" | tee "$STATE_DIR/last-entry-$name"
  echo "$stamp" > "$stamp_file"
}

entries=""
status=0
for name in index.sqlite code.sqlite; do
  if [ "$name" = code.sqlite ]; then
    if [ "$SYNC_CODE_INDEX" != 1 ] || [ ! -f "$INDEX_DIR/code.sqlite" ]; then
      # Code search turned off in DigUp: drop the Pi's copy too.
      remote "rm -f $REMOTE_DIR/code.sqlite" 2>>"$LOG_FILE" || true
      rm -f "$STATE_DIR"/last-*-code.sqlite
      continue
    fi
  fi
  if entry="$(sync_one "$name")"; then
    entries="${entries:+$entries, }$entry"
  else
    status=1
  fi
done

# The manifest says when the Pi's copy was last confirmed current, even when nothing changed.
if [ "$status" = 0 ]; then
  mkdir -p "$STAGING"
  cat > "$STAGING/manifest.json" <<EOF
{
  "synced_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "synced_at_epoch": $(date +%s),
  "source_host": "$(json_escape "$(scutil --get LocalHostName 2>/dev/null || hostname)")",
  "source_home": "$(json_escape "$HOME")",
  "digup_index_dir": "$(json_escape "$INDEX_DIR")",
  "files": {$entries}
}
EOF
  if rsync -t -e "${SSH_CMD[*]}" "$STAGING/manifest.json" "$PI_HOST:$PI_INDEX_DIR/manifest.json" \
       2>>"$LOG_FILE"; then
    [ "$VERBOSE" = 1 ] && log "manifest updated"
  else
    status=1
  fi
fi
exit "$status"
