import json
from typing import Any

import wandb

from ..evaluate import BaseTable
from .utils import to_jsonable


def _to_wandb_cell(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def log_results_wandb(tables: list[BaseTable]) -> None:
    # tables
    for table in tables:
        rows = table.records or []
        if not rows:
            continue

        cols = rows[0].keys()
        tbl = wandb.Table(columns=cols)
        for r in rows:
            tbl.add_data(*[to_jsonable(r.get(c)) for c in cols])  # pyright: ignore[reportUnknownMemberType]

        wandb.log({f"tables/{table.name}": tbl})
