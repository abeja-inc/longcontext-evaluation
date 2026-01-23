import json
from typing import Any

import wandb
from wandb.sdk.wandb_run import Run

from .data import Results
from .utils import iter_all_columns, to_jsonable


def _to_wandb_cell(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def push_results_to_wandb(*, results: Results, wandb_run: Any) -> None:
    run = wandb_run
    if not isinstance(run, Run):
        raise ValueError("wandb_run is not an instance of wandb.sdk.wandb_run.Run")

    # summary
    if results.summary:
        for k, v in results.summary.items():
            run.summary[k] = to_jsonable(v)

    # tables
    for table in results.tables:
        rows = table.rows or []
        if not rows:
            continue

        cols = iter_all_columns(rows)
        tbl = wandb.Table(columns=cols)
        for r in rows:
            tbl.add_data(*[_to_wandb_cell(r.get(c)) for c in cols])

        wandb.log({f"tables/{table.name}": tbl})
