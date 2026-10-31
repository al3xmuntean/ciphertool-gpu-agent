"""Resumable source uploads.

A source arrives in parts (`PUT /sources/{id}?offset=N`), each appended only at the current end of the partial file,
so a dropped connection resumes from `GET /sources/{id}`'s `received` without re-sending anything. Parts stay small
(tens of MB) because HTTPS proxies in front of rented GPUs cut long requests. `complete` checks size + SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable
from pathlib import Path

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class StorageError(Exception):
    def __init__(self, status: int, message: str, **extra: object) -> None:
        super().__init__(message)
        self.status = status
        self.extra = extra


def check_id(value: str) -> str:
    if not ID_RE.match(value or ""):
        raise StorageError(400, f"invalid id: {value!r}")
    return value


class SourceStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def _paths(self, source_id: str) -> tuple[Path, Path, Path]:
        check_id(source_id)
        return self.root / f"{source_id}.part", self.root / source_id, self.root / f"{source_id}.json"

    def path(self, source_id: str) -> Path:
        _, final, _ = self._paths(source_id)
        if not final.exists():
            raise StorageError(404, f"source {source_id} not uploaded")
        return final

    def status(self, source_id: str) -> dict:
        part, final, meta = self._paths(source_id)
        if final.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            return {"source_id": source_id, "complete": True, "received": info["size"], "sha256": info["sha256"]}
        return {"source_id": source_id, "complete": False, "received": part.stat().st_size if part.exists() else 0}

    def write_part(self, source_id: str, offset: int, chunks: Iterable[bytes]) -> int:
        part, final, _ = self._paths(source_id)
        with self._lock:
            if final.exists():
                raise StorageError(409, "source already complete", received=final.stat().st_size)
            received = part.stat().st_size if part.exists() else 0
            if offset != received:
                raise StorageError(409, f"offset {offset} != received {received}", received=received)
            with part.open("ab") as f:
                for chunk in chunks:
                    f.write(chunk)
            return part.stat().st_size

    def complete(self, source_id: str, size: int, sha256: str) -> dict:
        part, final, meta = self._paths(source_id)
        with self._lock:
            if final.exists():
                return self.status(source_id)
            if not part.exists() or part.stat().st_size != size:
                got = part.stat().st_size if part.exists() else 0
                raise StorageError(409, f"size mismatch: expected {size}, received {got}", received=got)
            digest = sha256_file(part)
            if digest.lower() != sha256.lower():
                part.unlink()
                raise StorageError(422, "sha256 mismatch, upload discarded", received=0)
            part.rename(final)
            meta.write_text(json.dumps({"size": size, "sha256": digest}), encoding="utf-8")
            return self.status(source_id)

    def delete(self, source_id: str) -> None:
        for p in self._paths(source_id):
            p.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
