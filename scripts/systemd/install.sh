#!/bin/sh
# Install/restart Engram as systemd user services (Linux). Logs: journalctl --user -u engram-api
# To keep it running while logged out: sudo loginctl enable-linger "$USER"
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEST="$HOME/.config/systemd/user"
mkdir -p "$DEST"
for SRC in "$REPO"/scripts/systemd/engram-*.service "$REPO"/scripts/systemd/engram-*.timer; do
  sed -e "s#__REPO__#$REPO#g" -e "s#__UV__#$(command -v uv)#g" "$SRC" > "$DEST/$(basename "$SRC")"
done
systemctl --user daemon-reload
systemctl --user enable --now engram-backup.timer
systemctl --user enable engram-api.service
systemctl --user restart engram-api.service
echo "engram-api: $(systemctl --user is-active engram-api) · backup timer: $(systemctl --user is-active engram-backup.timer)"
