import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Literal

from ..evaluate.table import BaseTable


def save_to_local(
    tables: list[BaseTable],
    output_root: Path,
    format: Literal["jsonl", "csv"] = "jsonl",
) -> None:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for table in tables:
        rows = table.rows or []
        if format == "jsonl":
            output_filepath = tables_dir / f"{table.name}.jsonl"
            with output_filepath.open("w", encoding="utf-8") as f:
                for row in rows:
                    json.dump(asdict(row), f, ensure_ascii=False)
                    f.write("\n")
        elif format == "csv":
            cols = [f.name for f in fields(rows[0])]
            output_filepath = tables_dir / f"{table.name}.csv"
            with output_filepath.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in rows:
                    w.writerow(asdict(row))
