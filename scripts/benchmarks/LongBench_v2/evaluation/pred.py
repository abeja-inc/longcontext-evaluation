import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import wandb

from benchmarks._base_benchmark.result import push_results_to_wandb
from benchmarks.longbench_v2.results import LongBenchResultsBuilder


# ---------- IO / UTILS ----------

def expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def _get_table(rows_by_name: dict[str, list[dict[str, Any]]], name: str) -> list[dict]:
    return rows_by_name.get(name, [])


def _index_tables(results) -> dict[str, list[dict[str, Any]]]:
    return {table.name: table.rows for table in results.tables}


# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--group", default="LongBench-v2")
    parser.add_argument("--compensate-missing", action="store_true")
    args = parser.parse_args()

    root = expand(args.results_dir) / args.model_name
    if not root.exists():
        print(f"[ERROR] No prediction files under: {root}")
        raise SystemExit(1)

    builder = LongBenchResultsBuilder(compensate_missing=args.compensate_missing)
    results = builder.build_results(
        model_name=args.model_name, prediction_dir=root, summary_json=None
    )
    tables = _index_tables(results)
    metrics_rows = _get_table(tables, "metrics")
    mean_rows = _get_table(tables, "metrics_mean")
    mean_metrics = mean_rows[0] if mean_rows else {}

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name or args.model_name,
        group=args.group,
        config={
            "model_name": args.model_name,
            "results_dir": str(root),
            "compensate_missing": args.compensate_missing,
        },
    )

    if push_results_to_wandb is None:
        raise RuntimeError("wandb is not available for logging results.")

    push_results_to_wandb(results=results, wandb_run=run)

    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mean_json_path = out_dir / f"{args.model_name}_metrics_mean.json"
    mean_json_path.parent.mkdir(parents=True, exist_ok=True)
    mean_json_path.write_text(
        json.dumps(mean_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if metrics_rows:
        met_cols = list(metrics_rows[0].keys())
        with (out_dir / f"{args.model_name}_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=met_cols)
            w.writeheader()
            w.writerows(metrics_rows)

    artifact_name = f"{args.model_name}-longbench-v2".replace("/", "_")
    art = wandb.Artifact(
        name=artifact_name,
        type="evaluation",
        metadata={"model": args.model_name},
    )
    art.add_file(str(args.output))
    if (out_dir / f"{args.model_name}_metrics.csv").exists():
        art.add_file(str(out_dir / f"{args.model_name}_metrics.csv"))
    if mean_json_path.exists():
        art.add_file(str(mean_json_path))
    wandb.log_artifact(art)

    print(f"Wrote {args.output} with {len(metrics_rows)} task rows.")
    wandb.finish()


if __name__ == "__main__":
    main()
