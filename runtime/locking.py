"""Single-instance protection for trading_runtime.py via a real OS-enforced
file lock — not a bare PID file. A PID file survives a crash (the PID it
names may no longer exist, or may have been reused by an unrelated
process) and can't answer "is the previous runtime actually still alive"
on its own; an OS lock is released automatically by Windows the instant
the holding process dies or is killed, so a stale lock is not possible.

Windows-only (this project runs on win32) — uses the stdlib `msvcrt`
module rather than a third-party cross-platform locking library, since a
single OS-specific primitive is all that's needed here.
"""

from __future__ import annotations

import msvcrt
from pathlib import Path
from typing import BinaryIO

_LOCK_BYTE_COUNT = 1


class RuntimeLockError(Exception):
    """Raised when another instance already holds the runtime lock."""


def acquire_lock(path: Path) -> BinaryIO:
    """Opens (creating if needed) and locks `path`. The returned handle
    must be kept open for the runtime's entire lifetime — closing it (or
    the process exiting/dying) releases the lock automatically. Call
    `release_lock` for a clean, explicit release on ordinary shutdown.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "r+b") if path.exists() else open(path, "w+b")
    handle.seek(0, 2)  # SEEK_END
    if handle.tell() == 0:
        # msvcrt.locking needs at least one real byte to lock — the
        # content itself is never read back; only its lock state matters.
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_BYTE_COUNT)
    except OSError as exc:
        handle.close()
        raise RuntimeLockError(
            f"Another instance already holds the runtime lock at {path} — refusing to start a second one."
        ) from exc
    return handle


def release_lock(handle: BinaryIO) -> None:
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTE_COUNT)
    finally:
        handle.close()
