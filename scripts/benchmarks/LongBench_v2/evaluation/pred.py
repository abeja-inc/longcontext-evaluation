import argparse
import csv
import json
import os
from pathlib import Path

import wandb
from project_module.benchmark.longbench_v2.scoring import LongBenchScorer

# ---------- IO / UTILS ----------
def expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def find_pred_files(root: Path) -> list[Path]:
    files: list[Path] = [path for path in root.rglob("*.jsonl")]
    return sorted(files)


def read_records(jsonl_filepath: Path) -> list[dict]:
    records = []
    with jsonl_filepath.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def to_wb_table(name: str, rows: list[dict]):
    if not rows:
        return None
    cols = list(rows[0].keys())
    tbl = wandb.Table(columns=cols)
    for r in rows:
        tbl.add_data(*[r.get(c) for c in cols])
    return name, tbl


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
    files = find_pred_files(root)
    if not files:
        print(f"[ERROR] No prediction files under: {root}")
        raise SystemExit(1)

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

    summary_rows: list[dict] = []
    metrics_rows: list[dict] = []

    scorer = LongBenchScorer(compensate_missing=args.compensate_missing)

    for filepath in files:
        recs = read_records(filepath)
        metrics, rows = scorer.score_records(recs)
        prompt_type = filepath.parent.name
        subset_name = filepath.stem

        # 集計行
        row = {
            "Model": args.model_name,
            "Prompt Type": prompt_type,
            "Subset": subset_name,
            **metrics,
        }
        metrics_rows.append(row)
        summary_rows.append(row)

        # サブセットごとの全予測をW&Bにログ
        if rows:
            subset_rows = [
                {
                    "Model": args.model_name,
                    "Prompt Type": prompt_type,
                    "Subset": subset_name,
                    **r,
                }
                for r in rows
            ]
            t = to_wb_table(f"predictions/{subset_name}", subset_rows)
            if t:
                wandb.log({t[0]: t[1]})

    # サブセット平均メトリクス
    accs = [r["overall_acc"] for r in summary_rows if r["overall_n"] > 0]
    run.summary["model/mean_overall_acc"] = (
        round(sum(accs) / len(accs), 2) if accs else 0.0
    )

    percent_keys = [
        "overall_acc",
        "easy_acc",
        "hard_acc",
        "short_acc",
        "medium_acc",
        "long_acc",
    ]
    mean_metrics = {"Model": args.model_name, "Prompt Type": "ALL", "Subset": "ALL"}
    mean_metrics["overall_n"] = sum(r["overall_n"] for r in summary_rows)
    mean_metrics["easy_n"] = sum(r["easy_n"] for r in summary_rows)
    mean_metrics["hard_n"] = sum(r["hard_n"] for r in summary_rows)
    mean_metrics["short_n"] = sum(r["short_n"] for r in summary_rows)
    mean_metrics["medium_n"] = sum(r["medium_n"] for r in summary_rows)
    mean_metrics["long_n"] = sum(r["long_n"] for r in summary_rows)
    for k in percent_keys:
        vals = [r[k] for r in summary_rows if r["overall_n"] > 0]
        mean_metrics[k] = round(sum(vals) / len(vals), 2) if vals else 0.0

    # W&B summary に平均値を記録
    for k in percent_keys:
        run.summary[f"mean/{k}"] = mean_metrics[k]
    run.summary["mean/overall_n"] = mean_metrics["overall_n"]

    # metrics テーブルをW&Bに送信
    metrics_table = to_wb_table("metrics/table", metrics_rows)
    if metrics_table:
        wandb.log({metrics_table[0]: metrics_table[1]})

    mean_tbl_cols = list(mean_metrics.keys())
    mean_tbl = wandb.Table(columns=mean_tbl_cols)
    mean_tbl.add_data(*[mean_metrics[c] for c in mean_tbl_cols])
    wandb.log({"metrics/mean": mean_tbl})

    # ローカル保存（JSONとCSVのみ）
    out_dir = args.output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
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

    # W&B Artifact 登録
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

    print(f"Wrote {args.output} with {len(summary_rows)} task rows.")
    wandb.finish()


if __name__ == "__main__":
    main()
