import csv
import json
from pathlib import Path
from typing import Literal

from .._evaluate import BaseTable
from .utils import to_jsonable


def log_results_local(
    tables: list[BaseTable],
    output_root: Path,
    format: Literal["jsonl", "csv"] = "jsonl",
) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for table in tables:
        rows = table.records or []
        cols = rows[0].keys()

        if format == "jsonl":
            output_filepath = tables_dir / f"{table.name}.jsonl"
            with output_filepath.open("w", encoding="utf-8") as f:
                for r in rows:
                    rr = {k: to_jsonable(v) for k, v in r.items()}
                    json.dump(rr, f, ensure_ascii=False)
                    f.write("\n")
        elif format == "csv":
            output_filepath = tables_dir / f"{table.name}.csv"
            with output_filepath.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({c: to_jsonable(r.get(c)) for c in cols})
