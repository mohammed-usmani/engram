#!/bin/sh
# Encrypted backup of everything Engram knows about you:
#   Postgres dump (memories, documents, queue) + context documents + mem0 history
# → one age-encrypted archive. Only the public key is needed here; restoring needs the private key.
#
#   scripts/backup.sh
#
# ENGRAM_BACKUP_DIR        where archives go   (default ~/dev/personal/engram-data/backups)
# ENGRAM_BACKUP_RECIPIENT  age public key file (default ~/.config/engram/backup.pub)
# ENGRAM_BACKUP_KEEP       archives to keep    (default 14)
# If the backup dir is inside a git repo, the new archive is committed and pushed (keep that repo PRIVATE).
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
env_get() { sed -n "s/^$1=\([^ #]*\).*/\1/p" "$REPO/.env" | tail -1; }

OUT=${ENGRAM_BACKUP_DIR:-$HOME/dev/personal/engram-data/backups}
RECIPIENT=${ENGRAM_BACKUP_RECIPIENT:-$HOME/.config/engram/backup.pub}
KEEP=${ENGRAM_BACKUP_KEEP:-14}
DB=$(env_get CONNECTION_STRING | sed 's/+asyncpg//')
DATA_DIR=$(env_get DATA_DIR); case "$DATA_DIR" in /*) ;; *) DATA_DIR="$REPO/$DATA_DIR" ;; esac
HISTORY=${MEM0_HISTORY_DB:-$HOME/.mem0/history.db}

[ -n "$DB" ] || { echo "CONNECTION_STRING missing from .env" >&2; exit 1; }
[ -f "$RECIPIENT" ] || { echo "no age public key at $RECIPIENT (age-keygen -o key; age-keygen -y key > $RECIPIENT)" >&2; exit 1; }
mkdir -p "$OUT"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

pg_dump --format=custom --no-owner --no-privileges "$DB" > "$TMP/engram.pgdump"
[ -d "$DATA_DIR" ] && cp -R "$DATA_DIR" "$TMP/contexts"
[ -f "$HISTORY" ] && cp "$HISTORY" "$TMP/mem0-history.db"
psql "$DB" -tAc "select 'episodes='||count(*) from episodes union all select 'facts='||count(*) from mem0_memories
                 union all select 'documents='||count(*) from context_documents" > "$TMP/MANIFEST" 2>/dev/null || true
date -u +%FT%TZ >> "$TMP/MANIFEST"

NAME="engram-$(date +%Y%m%d-%H%M%S).tar.age"
tar -C "$TMP" -czf - . | age -R "$RECIPIENT" > "$OUT/$NAME.part" && mv "$OUT/$NAME.part" "$OUT/$NAME"
echo "backup: $OUT/$NAME ($(du -h "$OUT/$NAME" | cut -f1)) — $(tr '\n' ' ' < "$TMP/MANIFEST")"

# keep the newest $KEEP archives
ls -1t "$OUT"/engram-*.tar.age 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r f; do rm -f "$f"; done

if git -C "$OUT" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$OUT" add -A . && git -C "$OUT" commit -qm "backup $NAME" && \
    { git -C "$OUT" push -q 2>/dev/null || echo "warning: push failed; archive kept locally" >&2; }
fi
