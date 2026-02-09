import json
from dataclasses import asdict, fields
from typing import Any, Sequence

import wandb

from ..evaluate.table import BaseTable, BaseTableRow


def _to_wandb_cell(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def push_to_wandb(tables: Sequence[BaseTable[BaseTableRow]]) -> None:
    for table in tables:
        rows = table.rows or []
        if not rows:
            continue

        cols: list[str] = [f.name for f in fields(rows[0])]
        wandb_table = wandb.Table(columns=cols)
        for row in rows:
            row_dict = asdict(row)
            wandb_table.add_data(*[_to_wandb_cell(row_dict.get(c)) for c in cols])  # pyright: ignore[reportUnknownMemberType]

        wandb.log({f"tables/{table.name}": wandb_table})
