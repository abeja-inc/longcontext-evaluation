import json
from typing import Any


def iter_all_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols = sorted({k for r in rows for k in r.keys()})
    return cols


def to_jsonable(v: Any) -> Any:
    # JSONL / CSV / W&B table で安全に扱える形へ
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)
