import argparse
import json
import logging
import re
import yaml
from pathlib import Path
from typing import Literal
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

from project_module.inference.config import GenerationConfig, OfflineServeConfig
from project_module.inference.io_data import Conversation, Message, Response
from project_module.inference.text_generator import OfflineTextGenerator
from project_module.utils import get_custom_logger


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

def get_n_tokens(tokenizer: AutoTokenizer, messages: list[Message], generation_config: GenerationConfig) -> int:
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        chat_template_kwargs=generation_config.chat_template_kwargs,
    )
    return len(token_ids)
    # return (sum([len(tokenizer.encode(message.content)) for message in messages]))

def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    model_path: Path = args.model_root.expanduser() / args.model_name
    dataset_dirpath: Path = args.dataset_dir.expanduser()
    output_dirpath: Path = args.output_dir.expanduser() / args.model_name

    with open(args.tasks, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        tasks: dict[Literal["mrcr"], list[str]] = raw["tasks"]
    
    with open(args.vllm_config, "r", encoding="utf-8") as f:
        vllm_config = yaml.safe_load(f)
    serve_cfg = OfflineServeConfig(model_name_or_path=str(model_path), **vllm_config["serve"])
    gen_cfg = GenerationConfig(**vllm_config["generation"])

    generator = OfflineTextGenerator(serve_config=serve_cfg, logger=logger)
    tokenizer = generator.llm.get_tokenizer()

    batch_size = args.batchsize

    for task_name in tasks:
        for dataset_filename in tasks[task_name]:
            pred_filepath = output_dirpath / task_name / dataset_filename
            pred_filepath.parent.mkdir(parents=True, exist_ok=True)
            pred_filepath.touch(exist_ok=False)

            dataset_filepath = dataset_dirpath / task_name / dataset_filename
            data = pd.read_parquet(dataset_filepath)

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
            data = data.drop(processed_ids, errors="ignore")
            logger.info(f"Skip {len(processed_ids)} already processed samples. Remaining: {len(data)}")

            total = len(data)
            for start in tqdm(
                range(0, total, batch_size),
                desc=f"Processing {task_name}/{dataset_filename}",
                total=int(total / batch_size),
            ):
                batch = data[start : start + batch_size]
                conversations: list[Conversation] = []
                conversations_to_generate: list[Conversation] = []
                target_context_lengths: list[int] = []
                # インプットデータの整形
                for _, sample in batch.iterrows():
                    conversation = Conversation(messages=[
                        Message(role=message["role"], content=message["content"])
                        for message in json.loads(sample["prompt"])
                    ])
                    n_tokens = get_n_tokens(tokenizer, conversation.messages, gen_cfg)
                    # コンテキスト長がmax_model_lenを超える場合は生成対象から除外
                    if n_tokens < serve_cfg.extra_args["max_model_len"]:
                        conversations_to_generate.append(conversation)
                    conversations.append(conversation)
                    target_context_lengths.append(n_tokens)

                # 推論
                if len(conversations_to_generate) == 0:
                    responses: list[Response] = []
                else:
                    responses: list[Response] = generator.generate(generation_config=gen_cfg, conversations=conversations_to_generate)

                # コンテキスト長がmax_model_lenを超える場合は生成結果を空文字列にする
                for idx in [i for i, v in enumerate(target_context_lengths) if v >= serve_cfg.extra_args["max_model_len"]]:
                    responses.insert(idx,
                        Response(prompt=conversations[idx], outputs=[""])
                    )

                with pred_filepath.open("a", encoding="utf-8") as f:
                    for (idx, sample), target_context_length, resp in zip(batch.iterrows(), target_context_lengths, responses, strict=True):
                        json.dump(
                            {
                                "id": idx,
                                "target_context_length": target_context_length,
                                "random_string_to_prepend": sample["random_string_to_prepend"],
                                "answer": sample["answer"],
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