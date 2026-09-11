"""Atomic artifact publication and exclusive, per-output run reservations.

Reservations protect cooperating repository commands, not unrelated programs.
A hard-killed process can leave a .lock file; never silently reclaim that lock.
"""

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import re
import tempfile


def validate_data_tag(tag):
    """Apply the providers' existing tag grammar before constructing paths."""
    if tag is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        raise ValueError("invalid --tag; use letters, digits, '.', '_' or '-'")
    return tag


def validate_run_label(label):
    if label is None:
        return None
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label)
            or label.upper().split(".")[0] in {
                "CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }):
        raise ValueError("--run-label must be 1-80 ASCII letters/digits/_/-, "
                         "start with a letter/digit, and not be a Windows device name")
    return label


@contextmanager
def reserve_artifacts(paths):
    """Fail before expensive work if any target exists or another run owns it."""
    targets = sorted({Path(path).expanduser().resolve() for path in paths}, key=str)
    if not targets:
        raise ValueError("at least one artifact target is required")
    locks = []
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite {target}. Use a new --run-label.")
            lock = target.with_name(f".{target.name}.lock")
            try:
                handle = lock.open("x", encoding="utf-8")
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Output reserved: {lock}. Choose a new --run-label; only remove "
                    "a stale lock after verifying its recorded PID has no active run.") from exc
            locks.append(lock)
            with handle:
                json.dump({"pid": os.getpid(), "target": str(target)}, handle)
        for target in targets:
            if target.exists():
                raise FileExistsError(f"Artifact appeared during reservation: {target}")
        yield
    finally:
        for lock in reversed(locks):
            lock.unlink(missing_ok=True)


@contextmanager
def atomic_open(path, mode="w", *, encoding="utf-8", newline=None):
    """Write beside the destination, then replace it only on successful close.

    Intended replacement is allowed (periodic checkpoints within a reserved run).
    An exception leaves an existing destination untouched and removes only this
    call's temporary file. This is per-file atomicity, not a multi-file transaction.
    """
    if mode not in {"w", "wb"}:
        raise ValueError("atomic_open only supports w or wb")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.",
                                     suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary)
    try:
        options = {} if "b" in mode else {"encoding": encoding, "newline": newline}
        with os.fdopen(fd, mode, **options) as handle:
            fd = None
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if fd is not None:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    with atomic_open(path) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to publish an empty result CSV")
    with atomic_open(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
