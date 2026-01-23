import csv
import json
from pathlib import Path
from typing import Any, Literal

from .data import Results
from .utils import iter_all_columns, to_jsonable


def save_results(
    results: Results, output_dirpath: Path, format: Literal["jsonl", "csv"] = "jsonl"
) -> Path:
    """
    output_dirpath に以下を保存:
      - manifest.json
      - tables/<table>.jsonl
      - tables/<table>.csv
    戻り値: manifest.json のパス
    """
    output_dirpath.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dirpath / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "summary": results.summary or {},
        "config": results.config or {},
        "tables": [],
    }

    for table in results.tables:
        rows = table.rows or []
        cols = iter_all_columns(rows) if rows else []

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

        manifest["tables"].append(
            {
                "name": table.name,
                "rows": len(rows),
                "path": str(output_filepath.relative_to(output_dirpath)),
            }
        )

    manifest_path = output_dirpath / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path
