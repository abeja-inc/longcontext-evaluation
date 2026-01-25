import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb
import yaml

from project_module.benchmark.mrcr.evaluation import EvaluationConfig, EvaluationPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=Path("./eval_config.yaml"), help="Path to YAML config file"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/longcontext-llm-benchmark-eval/experimentation/outputs"),
        help="Path to output directory",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--group", type=str, default="MRCR")
    p.add_argument("--model-name", type=str, required=True, help="model name")
    return p.parse_args()

def _flatten_rows(model_name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for task_obj in summary.get("results", []):
        task = task_obj.get("task")
        for subset in task_obj.get("subsets", []):
            subset_name = subset.get("subset_name")
            for s in subset.get("score", []):
                flat.append(
                    {
                        "Model": model_name,
                        "Task": task,
                        "Subset": subset_name,
                        "context_length": s.get("context_length"),
                        "score": s.get("score"),
                    }
                )
    return flat


def _mean_score(rows: list[dict[str, Any]]) -> float:
    vals = [r["score"] for r in rows if isinstance(r.get("score"), (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0

def _wandb_log(summary_json: dict[str, Any], model_name: str, cfg_for_run: dict[str, Any]) -> None:
    project = cfg_for_run.get("wandb_project")
    entity = cfg_for_run.get("wandb_entity")
    run_name = cfg_for_run.get("run_name") or model_name
    group = cfg_for_run.get("group") or "MRCR"

    if not project:
        return

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        config={
            "model_name": model_name,
            "output_dir": str(cfg_for_run.get("output_dir")),
            "config_path": str(cfg_for_run.get("config_path")),
        },
    )

    # ===== 行データ化 & テーブル =====
    flat = _flatten_rows(model_name, summary_json)  # 行 = (Task, Subset, context_length, score)
    if flat:
        cols = list(flat[0].keys())
        tbl = wandb.Table(columns=cols)
        for r in flat:
            tbl.add_data(*[r.get(c) for c in cols])
        wandb.log({"metrics/by_subset_ctx": tbl})

    # ===== タスク別の平均（Subset×Context の単純平均） =====
    mean_rows = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in flat:
        by_task[r["Task"]].append(r)

    for task, rows in by_task.items():
        mean_rows.append({"Model": model_name, "Task": task, "mean_score": _mean_score(rows)})

    # 全体平均（全行の単純平均）
    overall_mean = _mean_score(flat)
    run.summary["mean/overall_score"] = overall_mean
    for mr in mean_rows:
        run.summary[f"mean/{mr['Task']}"] = mr["mean_score"]

    # テーブル（タスク平均の比較用）
    if mean_rows:
        cols2 = list(mean_rows[0].keys())
        mean_tbl = wandb.Table(columns=cols2)
        for r in mean_rows:
            mean_tbl.add_data(*[r.get(c) for c in cols2])
        wandb.log({"metrics/task_means": mean_tbl})

    # summary.json を Artifact で保存
    artifact_name = f"{args.model_name}-mrcr-eval".replace("/", "_")
    art = wandb.Artifact(
        name=artifact_name,
        type="evaluation",
        metadata={"model": model_name, "kind": "MRCR"},
    )
    # このファイルは run.py の外で作られているため、パスは config から再構築する
    summary_path = Path(cfg_for_run["output_filepath"])
    if summary_path.exists():
        art.add_file(str(summary_path))
        wandb.log_artifact(art)

    # ===== Artifact: 生予測（prediction_dirpath配下の *.jsonl を全登録） =====
    pred_dir = Path(cfg_for_run["prediction_dirpath"])
    if pred_dir.exists():
        jsonl_files = sorted(pred_dir.rglob("*.jsonl"))
        if jsonl_files:
            artifact_name = f"{args.model_name}-mrcr-preds".replace("/", "_")
            preds_art = wandb.Artifact(
                name=artifact_name,
                type="predictions",
                metadata={"model": model_name, "kind": "MRCR", "count": len(jsonl_files)},
            )
            for fp in jsonl_files:
                # Artifact内のパスは prediction_dirpath からの相対で保存
                preds_art.add_file(str(fp), name=str(fp.relative_to(pred_dir)))
            wandb.log_artifact(preds_art)

    wandb.finish()


def main(args: argparse.Namespace):
    # 評価設定ファイル（例: config.json）を読み込む
    with args.config.open("r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)

    output_dirpath = args.output_dir.expanduser()
    model_name = args.model_name

    prediction_dirpath = output_dirpath / model_name
    output_filepath = prediction_dirpath / "summary.json"

    config_dict = {
        "prediction_dirpath": prediction_dirpath,
        "tasks": eval_config["tasks"],
    }
    config = EvaluationConfig(**config_dict)

    pipeline = EvaluationPipeline(output_filepath=output_filepath)
    pipeline.run(config)

    cfg_for_run = {
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "run_name": args.run_name,
        "group": args.group,
        "output_dir": str(output_dirpath),
        "config_path": str(args.config),
        "output_filepath": str(output_filepath),
        "prediction_dirpath": str(prediction_dirpath),  # ← これを追加
    }

    try:
        with output_filepath.open("r", encoding="utf-8") as f:
            summary_json = json.load(f)
    except Exception:
        return

    _wandb_log(summary_json=summary_json, model_name=model_name, cfg_for_run=cfg_for_run)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
