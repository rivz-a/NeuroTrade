"""Single-instance protection for trading_runtime.py via a real OS-enforced
file lock — not a bare PID file. A PID file survives a crash (the PID it
names may no longer exist, or may have been reused by an unrelated
process) and can't answer "is the previous runtime actually still alive"
on its own; an OS lock is released automatically the instant the holding
process dies or is killed, so a stale lock is not possible.

Cross-platform: uses `msvcrt` on Windows and `fcntl` on POSIX (Linux/macOS,
i.e. the VDS this runs on in production) — both are stdlib, so no
third-party locking library is needed for this single primitive.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import BinaryIO

_LOCK_BYTE_COUNT = 1


class RuntimeLockError(Exception):
    """Raised when another instance already holds the runtime lock."""


if sys.platform == "win32":
    import msvcrt

    def _lock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_BYTE_COUNT)

    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTE_COUNT)
else:
    import fcntl

    def _lock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
        _lock(handle)
    except OSError as exc:
        handle.close()
        raise RuntimeLockError(
            f"Another instance already holds the runtime lock at {path} — refusing to start a second one."
        ) from exc
    return handle


def release_lock(handle: BinaryIO) -> None:
    try:
        _unlock(handle)
    finally:
        handle.close()
