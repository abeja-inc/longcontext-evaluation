import argparse
import json
import logging
import re
from pathlib import Path
from typing import Literal

import yaml
from project_module.inference.config import GenerationConfig, OfflineServeConfig
from project_module.inference.io_data import Response
from project_module.inference.text_generator import OfflineTextGenerator
from project_module.utils import get_custom_logger
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=Path, required=True, help="Path to models root directory")
    p.add_argument("--model-name", type=str, required=True, help="Name of the model")
    p.add_argument(
        "--vllm-config",
        type=Path,
        default=Path("./vllm_offline_config.yml"),
        help="Path to vllm config",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/longcontext-llm-benchmark-eval/experimentation/outputs"),
        help="Path to output directory",
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("~/datasets"), help="Path to dataset directory"
    )
    p.add_argument("--tasks", type=Path, default=Path("./tasks.yml"), help="Path to tasks file")
    p.add_argument("--batchsize", type=int, default=1, help="Batch size for inference")
    return p.parse_args()


def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    model_path: Path = args.model_root.expanduser() / args.model_name
    dataset_dirpath: Path = args.dataset_dir.expanduser()
    output_dirpath: Path = args.output_dir.expanduser() / args.model_name

    with open(args.tasks, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        tasks: dict[Literal["niah", "qa"], list[str]] = raw["tasks"]

    with open(args.vllm_config, "r", encoding="utf-8") as f:
        vllm_config = yaml.safe_load(f)
    serve_cfg = OfflineServeConfig(model_name_or_path=str(model_path), **vllm_config["serve"])
    gen_cfg = GenerationConfig(**vllm_config["generation"])

    generator = OfflineTextGenerator(serve_config=serve_cfg, logger=logger)
    tokenizer = generator.llm.get_tokenizer()

    batch_size = args.batchsize

    for task_name in tasks:
        for dataset_filename in tasks[task_name]:
            # 出力パスを設定
            pred_filepath = output_dirpath / task_name / dataset_filename
            pred_filepath.parent.mkdir(parents=True, exist_ok=True)
            pred_filepath.touch()

            # データセットをロード
            dataset_filepath = dataset_dirpath / task_name / dataset_filename
            with dataset_filepath.open("r") as f:
                data = [json.loads(line) for line in f]

            # 処理済みのサンプルがある場合はスキップ
            processed_ids = set()
            if pred_filepath.exists():
                with pred_filepath.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line)
                                processed_ids.add(record["id"])
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON in {pred_filepath}")

            data = [sample for sample in data if sample["sample_id"] not in processed_ids]
            logger.info(
                f"Skip {len(processed_ids)} already processed samples. Remaining: {len(data)}"
            )

            # 推論
            total = len(data)
            for start in tqdm(
                range(0, total, batch_size),
                desc=f"Processing {task_name}/{dataset_filename}",
                total=int(total / batch_size),
            ):
                batch = data[start : start + batch_size]
                conversations = [
                    [{"role": "user", "content": sample["content"]["user_prompt"]}]
                    for sample in batch
                ]

                # answer prefix を指定するために手動でチャットテンプレートを適用
                prompts = tokenizer.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=gen_cfg.chat_template_kwargs["enable_thinking"],
                )
                prompts = [
                    prompt + sample["content"]["answer_prefix"]
                    for prompt, sample in zip(prompts, batch, strict=True)
                ]

                # NOTE:
                #     answer_prefix を追加する必要がない場合は、以下のようなデータ型への変換をすると、
                #     generator 内でデフォルトのチャットテンプレートが適用される。
                #     prompts で入力すると、内部ではチャットテンプレートは適用されず、そのまま推論が行われる。
                #     """
                #     conversations = [
                #         Conversation.model_validate([{"role": "user", "content": sample["content"]["user_prompt"]}])
                #         for sample in batch
                #     ]
                #     responses: list[Response] = generator.generate(generation_config=gen_cfg, conversations=conversations)
                #     """
                responses: list[Response] = generator.generate(
                    generation_config=gen_cfg, prompts=prompts
                )

                # 出力を保存
                with pred_filepath.open("a", encoding="utf-8") as f:
                    for sample, resp in zip(batch, responses, strict=True):
                        json.dump(
                            {
                                "id": sample["sample_id"],
                                "target_context_length": sample["target_context_length"],
                                "answer": sample["content"]["outputs"],
                                "prediction": resp.outputs[0],
                            },
                            f,
                            ensure_ascii=False,
                        )
                        f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    logger = get_custom_logger()
    main(args=args, logger=logger)
