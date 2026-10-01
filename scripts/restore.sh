#!/bin/sh
# Restore an Engram backup (from scripts/backup.sh) — e.g. onto a new machine.
#
#   scripts/restore.sh ARCHIVE.tar.age --db postgresql://user:pass@host:5432/dbname [--contexts DIR] [--force]
#
# --db        target database (created if missing). Refuses a database that already holds memories
#             unless --force is given — restoring REPLACES its contents.
# --contexts  also restore the context documents into DIR (then point DATA_DIR there)
# Needs the private key: ENGRAM_BACKUP_KEY (default ~/.config/engram/backup.key).
set -eu
ARCHIVE=${1:?usage: restore.sh ARCHIVE --db URL [--contexts DIR] [--force]}; shift
DB="" CONTEXTS="" FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --db) DB=$2; shift 2 ;;
    --contexts) CONTEXTS=$2; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$DB" ] || { echo "--db is required" >&2; exit 2; }
KEY=${ENGRAM_BACKUP_KEY:-$HOME/.config/engram/backup.key}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

age -d -i "$KEY" "$ARCHIVE" | tar -C "$TMP" -xzf -
echo "archive: $(tr '\n' ' ' < "$TMP/MANIFEST")"

DBNAME=${DB##*/}; DBNAME=${DBNAME%%\?*}
ADMIN=${DB%/*}/postgres
if ! psql "$DB" -tAc "select 1" >/dev/null 2>&1; then
  psql "$ADMIN" -qc "create database \"$DBNAME\"" && echo "created database $DBNAME"
fi
EXISTING=$(psql "$DB" -tAc "select coalesce((select count(*) from episodes),0) + coalesce((select count(*) from context_documents),0)" 2>/dev/null || echo 0)
if [ "${EXISTING:-0}" -gt 0 ] && [ "$FORCE" -ne 1 ]; then
  echo "refusing: $DBNAME already holds $EXISTING rows of memory; re-run with --force to replace it" >&2
  exit 1
fi
psql "$DB" -qc "create extension if not exists vector" 2>/dev/null || true
pg_restore --clean --if-exists --no-owner --no-privileges -d "$DB" "$TMP/engram.pgdump" 2>&1 \
  | grep -vE "does not exist, skipping|transaction_timeout|errors ignored on restore" >&2 || true
echo "restored: $(psql "$DB" -tAc "select 'episodes='||count(*) from episodes union all select 'facts='||count(*) from mem0_memories union all select 'documents='||count(*) from context_documents" | tr '\n' ' ')"

if [ -n "$CONTEXTS" ] && [ -d "$TMP/contexts" ]; then
  mkdir -p "$CONTEXTS" && cp -R "$TMP/contexts/." "$CONTEXTS/" && echo "contexts → $CONTEXTS"
fi
HISTORY=${MEM0_HISTORY_DB:-$HOME/.mem0/history.db}
if [ -f "$TMP/mem0-history.db" ] && { [ ! -f "$HISTORY" ] || [ "$FORCE" -eq 1 ]; }; then
  mkdir -p "$(dirname "$HISTORY")" && cp "$TMP/mem0-history.db" "$HISTORY" && echo "mem0 history → $HISTORY"
fi
