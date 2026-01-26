import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    dataset: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                dataset.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return dataset


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def load_processed_ids(path: Path, *, id_key: str = "id") -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
                if id_key in r:
                    ids.add(str(r[id_key]))
            except Exception:
                continue
    return ids
