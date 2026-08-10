#!/usr/bin/env bash
# Backs up journal.db via SQLite's own .backup command (safe to run against a
# live WAL-mode DB — unlike `cp`, it doesn't risk copying a torn/inconsistent
# file mid-write). Keeps the last KEEP_COUNT backups, deletes the rest.
set -euo pipefail

APP_DIR="/opt/neurotrade"
BACKUP_DIR="${APP_DIR}/backups"
KEEP_COUNT=48

mkdir -p "$BACKUP_DIR"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
dest="${BACKUP_DIR}/journal-${timestamp}.db"

sqlite3 "${APP_DIR}/journal.db" ".backup '${dest}'"
gzip "$dest"

ls -1t "${BACKUP_DIR}"/journal-*.db.gz 2>/dev/null | tail -n "+$((KEEP_COUNT + 1))" | xargs -r rm --

echo "Backup complete: ${dest}.gz"
