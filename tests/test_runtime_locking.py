"""Offline tests for runtime/locking.py — no network, no subprocess. Uses
real OS file locks (msvcrt on Windows, fcntl on POSIX) against a pytest
tmp_path.
"""

from __future__ import annotations

import pytest

from runtime.locking import RuntimeLockError, acquire_lock, release_lock


def test_acquire_and_release_lock(tmp_path):
    lock_path = tmp_path / "runtime.lock"
    handle = acquire_lock(lock_path)
    assert lock_path.exists()
    release_lock(handle)


def test_second_acquire_while_held_raises(tmp_path):
    lock_path = tmp_path / "runtime.lock"
    handle = acquire_lock(lock_path)
    try:
        with pytest.raises(RuntimeLockError):
            acquire_lock(lock_path)
    finally:
        release_lock(handle)


def test_acquire_again_after_release_succeeds(tmp_path):
    lock_path = tmp_path / "runtime.lock"
    first = acquire_lock(lock_path)
    release_lock(first)

    second = acquire_lock(lock_path)
    release_lock(second)


def test_acquire_creates_parent_directory(tmp_path):
    lock_path = tmp_path / "nested" / "dir" / "runtime.lock"
    handle = acquire_lock(lock_path)
    try:
        assert lock_path.exists()
    finally:
        release_lock(handle)
