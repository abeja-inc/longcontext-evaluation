import json
from dataclasses import asdict
from typing import Any

import wandb

from ..evaluate import BaseTable


def _to_wandb_cell(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def push_to_wandb(tables: list[BaseTable]) -> None:
    # tables
    for table in tables:
        rows = table.rows or []
        if not rows:
            continue

        cols = rows[0].keys()
        tbl = wandb.Table(columns=cols)
        for row in rows:
            row_dict = asdict(row)
            tbl.add_data(*[_to_wandb_cell(row_dict.get(c)) for c in cols])  # pyright: ignore[reportUnknownMemberType]

        wandb.log({f"tables/{table.name}": tbl})
