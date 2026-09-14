import json
import os
from pathlib import Path

from evals_repro.cache import digest


def fingerprint(value) -> str:
    return digest([json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)])


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("rb+") as stream:
        while line := stream.readline():
            end = stream.tell()
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError(f"Corrupt record in {path} at byte {end - len(line)}") from None
                stream.truncate(end - len(line))
                break
            rows.append(row)
            if not line.endswith(b"\n"):
                stream.write(b"\n")
    return rows


def append_record(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
