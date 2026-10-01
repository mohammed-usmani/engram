#!/bin/sh
# Install/restart Engram's macOS login services:
#   com.engram.api     — REST + MCP (/mcp) + worker on 127.0.0.1:8001   (logs ~/Library/Logs/engram.log)
#   com.engram.backup  — nightly encrypted backup at 02:30             (logs ~/Library/Logs/engram-backup.log)
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
UID_=$(id -u)
# remove the pre-rename service if it is still around
launchctl bootout "gui/$UID_/com.personalcontext.api" 2>/dev/null && rm -f "$HOME/Library/LaunchAgents/com.personalcontext.api.plist" || true
for SRC in "$REPO"/scripts/launchd/com.engram.*.plist; do
  LABEL=$(basename "$SRC" .plist)
  DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
  sed -e "s#__REPO__#$REPO#g" -e "s#__UV__#$(command -v uv)#" -e "s#__HOME__#$HOME#g" "$SRC" > "$DEST"
  launchctl bootout "gui/$UID_/$LABEL" 2>/dev/null || true
  # bootout returns before the job is gone; bootstrapping too early fails with "Input/output error"
  for _ in 1 2 3 4 5 6 7 8 9 10; do launchctl print "gui/$UID_/$LABEL" >/dev/null 2>&1 || break; sleep 1; done
  launchctl bootstrap "gui/$UID_" "$DEST"
  echo "installed: $LABEL"
done
