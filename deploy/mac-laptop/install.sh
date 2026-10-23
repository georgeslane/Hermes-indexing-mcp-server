#!/bin/bash
# Sets up the index sync on the Mac that runs DigUp (Mac no.1): a launchd job that runs digup-sync.sh
# every few minutes, and again whenever the Mac wakes and the interval has passed.
#
#   deploy/mac-laptop/install.sh --pi pi@raspberrypi [--interval 900] [--key ~/.ssh/hermes_digup]
#   deploy/mac-laptop/install.sh --uninstall

set -euo pipefail

LABEL="com.hermes-digup.sync"
HERE="$(cd "$(dirname "$0")" && pwd)"
STATE_DIR="$HOME/Library/Application Support/hermes-digup"
CONFIG_DIR="$HOME/.config/hermes-digup"
CONFIG="$CONFIG_DIR/sync.env"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SCRIPT="$STATE_DIR/digup-sync.sh"
PI=""
INTERVAL=900
KEY=""
UNINSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --pi) PI="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --key) KEY="$2"; shift 2 ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ "$(uname -s)" = Darwin ] || { echo "This is for the Mac that runs DigUp." >&2; exit 1; }

if [ "$UNINSTALL" = 1 ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST" "$SCRIPT"
  echo "Removed the sync job. Settings are still in $CONFIG."
  exit 0
fi

mkdir -p "$STATE_DIR" "$CONFIG_DIR" "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
install -m 755 "$HERE/digup-sync.sh" "$SCRIPT"

if [ ! -f "$CONFIG" ]; then
  if [ -z "$PI" ]; then
    read -r -p "SSH login for the Pi over Tailscale (e.g. pi@raspberrypi): " PI
  fi
  [ -n "$PI" ] || { echo "Need the Pi's user@host." >&2; exit 1; }
  cat > "$CONFIG" <<EOF
# Settings for digup-sync.sh (the DigUp index -> Raspberry Pi copy).
PI_HOST=$PI
# Where the Pi keeps its copy (relative to the Pi user's home). Must match DIGUP_INDEX_DIR on the Pi.
PI_INDEX_DIR=.local/share/hermes-digup/index
# A dedicated SSH key, if you made one (no spaces in the path).
SSH_KEY=${KEY}
# Only if you moved DigUp's index somewhere else (DigUp > Settings shows where it is):
#DIGUP_INDEX_DIR=
# 1 = don't sync while on battery.
SKIP_ON_BATTERY=0
# 1 = also copy DigUp's code index (code.sqlite) when code search is on.
SYNC_CODE_INDEX=1
EOF
  echo "Wrote $CONFIG"
else
  echo "Keeping existing $CONFIG"
  if [ -n "$PI" ]; then sed -i '' "s|^PI_HOST=.*|PI_HOST=$PI|" "$CONFIG"; fi
  if [ -n "$KEY" ]; then sed -i '' "s|^SSH_KEY=.*|SSH_KEY=$KEY|" "$CONFIG"; fi
fi

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$SCRIPT</string></array>
  <key>StartInterval</key><integer>$INTERVAL</integer>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>Nice</key><integer>10</integer>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/hermes-digup-sync.launchd.log</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/hermes-digup-sync.launchd.log</string>
</dict>
</plist>
EOF
plutil -lint "$PLIST" >/dev/null

# shellcheck source=/dev/null
. "$CONFIG"
echo "Checking SSH to $PI_HOST (no password prompts allowed: launchd can't answer them)..."
SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=10)
[ -n "${SSH_KEY:-}" ] && SSH_ARGS+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
if ssh "${SSH_ARGS[@]}" "$PI_HOST" "command -v rsync >/dev/null || echo 'rsync is missing on the Pi: sudo apt install rsync'"; then
  echo "SSH works."
else
  cat <<EOF

SSH to $PI_HOST didn't work without a password. Set up a key once, then run this again:
  ssh-keygen -t ed25519 -f ~/.ssh/hermes_digup -N ""
  ssh-copy-id -i ~/.ssh/hermes_digup.pub $PI_HOST
  $0 --key ~/.ssh/hermes_digup
EOF
  exit 1
fi

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed: syncing every $((INTERVAL / 60)) minutes (first run starting now)."
echo "Log: ~/Library/Logs/hermes-digup-sync.log    Run by hand: \"$SCRIPT\" --force -v"
