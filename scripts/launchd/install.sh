#!/bin/sh
# Install/restart Engram's macOS login services:
#   com.engram.api     — REST + MCP (/mcp) + worker on 127.0.0.1:8001   (logs ~/Library/Logs/engram.log)
#   com.engram.backup  — nightly encrypted backup at 02:30             (logs ~/Library/Logs/engram-backup.log)
#   com.engram.sync    — save assistant conversations every 10 min     (logs ~/Library/Logs/engram-sync.log)
#
# Re-registers a service only when its plist changed: every registration makes macOS show an
# "App Background Activity" notification. Otherwise the API is just restarted in place.
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
# first install: mark existing transcripts as read so years of history aren't sent at once
[ -f "$HOME/.config/engram/sync-offsets.json" ] || /usr/bin/python3 "$REPO/scripts/transcript_sync.py" --init
UID_=$(id -u)
# remove the pre-rename service if it is still around
launchctl bootout "gui/$UID_/com.personalcontext.api" 2>/dev/null && rm -f "$HOME/Library/LaunchAgents/com.personalcontext.api.plist" || true
for SRC in "$REPO"/scripts/launchd/com.engram.*.plist; do
  LABEL=$(basename "$SRC" .plist)
  DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
  NEW=$(sed -e "s#__REPO__#$REPO#g" -e "s#__UV__#$(command -v uv)#" -e "s#__HOME__#$HOME#g" "$SRC")
  if [ -f "$DEST" ] && [ "$NEW" = "$(cat "$DEST")" ] && launchctl print "gui/$UID_/$LABEL" >/dev/null 2>&1; then
    case "$LABEL" in
      com.engram.api) launchctl kickstart -k "gui/$UID_/$LABEL" && echo "restarted: $LABEL" ;;
      *) echo "unchanged: $LABEL" ;;
    esac
    continue
  fi
  printf '%s\n' "$NEW" > "$DEST"
  launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
  # bootout returns before the job is gone; bootstrapping too early fails with "Input/output error"
  for _ in 1 2 3 4 5 6 7 8 9 10; do launchctl print "gui/$UID_/$LABEL" >/dev/null 2>&1 || break; sleep 1; done
  launchctl bootstrap "gui/$UID_" "$DEST"
  echo "installed: $LABEL"
done
