import csv
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal, Sequence

from ..evaluate.table import BaseTable, BaseTableRow


def _to_jsonable_row(row: BaseTableRow) -> dict[str, Any]:
    row_dict = asdict(row)
    return json.loads(json.dumps(row_dict, ensure_ascii=False, default=str))


def _to_csv_cell(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def save_to_local(
    tables: Sequence[BaseTable[BaseTableRow]],
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
                    json.dump(_to_jsonable_row(row), f, ensure_ascii=False)
                    f.write("\n")
        elif format == "csv":
            output_filepath = tables_dir / f"{table.name}.csv"
            if not rows:
                output_filepath.touch()
                continue
            cols = [f.name for f in fields(rows[0])]
            with output_filepath.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in rows:
                    row_dict = asdict(row)
                    w.writerow({c: _to_csv_cell(row_dict.get(c)) for c in cols})
