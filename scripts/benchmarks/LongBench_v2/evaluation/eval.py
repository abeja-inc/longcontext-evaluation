import argparse
import json
import logging
import os
from pathlib import Path

import yaml
from project_module.inference.config import GenerationConfig, OfflineServeConfig
from project_module.inference.io_data import Conversation, Response
from project_module.inference.text_generator import OfflineTextGenerator
from project_module.utils import get_custom_logger
from tqdm import tqdm
from transformers import AutoTokenizer


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-root", type=Path, required=True, help="Path to models root directory"
    )
    p.add_argument("--model-name", type=str, required=True, help="model name")
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
        "--dataset-dir",
        type=Path,
        default=Path("~/datasets"),
        help="Path to dataset directory",
    )
    p.add_argument(
        "--tasks", type=Path, default=Path("./tasks.yml"), help="Path to tasks file"
    )
    p.add_argument(
        "--rag-topn",
        type=int,
        default=0,
        help="Use top-N retrieved chunks if dataset has them",
    )
    p.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    p.add_argument("--no-context", action="store_true", help="Do not use context")
    p.add_argument("--batchsize", type=int, default=1, help="Batch size for inference")
    return p.parse_args()


def load_prompts(tasks_yml: Path) -> dict[str, str]:
    with open(tasks_yml, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    prompt_dir = expand_path(raw["prompt"]["dirpath"])
    mapping: dict[str, str] = raw["prompt"]["prompts"]

    prompts: dict[str, str] = {}
    for prompt_key, prompt_file_name in mapping.items():
        prompt_file = prompt_dir / prompt_file_name
        prompts[prompt_key] = prompt_file.read_text(encoding="utf-8")

    return prompts


def build_prompt_from_template(
    prompt_templates: dict[str, str],
    sample: dict,
    rag_topn: int = 0,
    cot: bool = False,
    no_context: bool = False,
) -> tuple[str, str]:
    # DOC
    if rag_topn > 0 and "retrieved_context" in sample:
        template = prompt_templates["0shot_rag"]
        chunks = sorted(sample["retrieved_context"], key=lambda x: x.get("c_idx", 0))[
            :rag_topn
        ]
        doc = "\n\n".join(
            [
                f"Retrieved chunk {i + 1}: {c.get('content', '')}"
                for i, c in enumerate(chunks)
            ]
        )
    else:
        doc = sample["context"]
        if no_context:
            template = prompt_templates["0shot_no_context"]
        elif cot:
            template = prompt_templates["0shot_cot"]
        else:
            template = prompt_templates["0shot"]

    # Question & choices
    q = sample["question"]
    c_a = sample["choice_A"]
    c_b = sample["choice_B"]
    c_c = sample["choice_C"]
    c_d = sample["choice_D"]

    # Build prompt
    filled = (
        template.replace("$DOC$", doc.strip())
        .replace("$Q$", q.strip())
        .replace("$C_A$", c_a.strip())
        .replace("$C_B$", c_b.strip())
        .replace("$C_C$", c_c.strip())
        .replace("$C_D$", c_d.strip())
    )

    # extract answer template
    if cot:
        answer_template = prompt_templates["0shot_cot_ans"]
        answer_template = (
            answer_template.replace("$DOC$", doc.strip())
            .replace("$Q$", q.strip())
            .replace("$C_A$", c_a.strip())
            .replace("$C_B$", c_b.strip())
            .replace("$C_C$", c_c.strip())
            .replace("$C_D$", c_d.strip())
        )
    else:
        answer_template = ""
    return filled, answer_template


def truncate(
    content: str,
    tokenizer: AutoTokenizer,
    max_model_len: int,
    max_new_token: int,
    buffer: int = 30,
) -> str:
    max_len = max_model_len - max_new_token - buffer

    input_ids = tokenizer.encode(content)
    if len(input_ids) > max_len:
        input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
        truncated_content = tokenizer.decode(input_ids, skip_special_tokens=True)
        return truncated_content
    else:
        return content


def main(args: argparse.Namespace, logger: logging.Logger) -> None:
    model_path: Path = expand_path(args.model_root) / args.model_name
    dataset_dirpath: Path = expand_path(args.dataset_dir)
    output_dirpath: Path = expand_path(args.output_dir) / args.model_name

    prompt_templates = load_prompts(args.tasks)

    with open(args.vllm_config, "r", encoding="utf-8") as f:
        vllm_config = yaml.safe_load(f)
    serve_cfg = OfflineServeConfig(
        model_name_or_path=str(model_path), **vllm_config["serve"]
    )
    gen_cfg = GenerationConfig(**vllm_config["generation"])

    generator = OfflineTextGenerator(serve_config=serve_cfg, logger=logger)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    max_model_len = vllm_config["serve"]["extra_args"]["max_model_len"]
    max_new_token = vllm_config["generation"]["max_new_tokens"]

    batch_size = args.batchsize

    if args.rag_topn > 0:
        subdir_name = "0shot_rag"
    elif args.cot:
        subdir_name = "0shot_cot"
    elif args.no_context:
        subdir_name = "0shot_no_context"
    else:
        subdir_name = "0shot"

    for dataset_filepath in dataset_dirpath.rglob("*_with_token_count.jsonl"):
        pred_filepath = output_dirpath / subdir_name / f"{dataset_filepath.stem}.jsonl"
        pred_filepath.parent.mkdir(parents=True, exist_ok=True)

        # データセットをロード
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
        else:
            pred_filepath.touch()

        data = [sample for sample in data if sample["sample_id"] not in processed_ids]
        logger.info(
            f"Skip {len(processed_ids)} already processed samples. Remaining: {len(data)}"
        )

        # 推論
        total = len(data)
        for start in tqdm(
            range(0, total, batch_size),
            desc=f"Processing {dataset_filepath.name}",
            total=int(total / batch_size),
        ):
            batch = data[start : start + batch_size]
            conversations: list[Conversation] = []
            for sample in batch:
                user_prompt, answer_template = build_prompt_from_template(
                    prompt_templates=prompt_templates,
                    sample=sample,
                    rag_topn=args.rag_topn,
                    cot=args.cot,
                    no_context=args.no_context,
                )
                truncated_user_prompt = truncate(
                    content=user_prompt,
                    tokenizer=tokenizer,
                    max_model_len=max_model_len,
                    max_new_token=max_new_token,
                )
                conversations.append(
                    Conversation(
                        messages=[{"role": "user", "content": truncated_user_prompt}]
                    )
                )
            responses: list[Response] = generator.generate(
                generation_config=gen_cfg, conversations=conversations
            )

            if args.cot:
                conversations: list[Conversation] = []
                for resp in responses:
                    conversations.append(
                        Conversation(
                            messages=[
                                {
                                    "role": "assistant",
                                    "content": answer_template.replace(
                                        "$COT$",
                                        truncate(
                                            content=resp.outputs[0].strip(),
                                            tokenizer=tokenizer,
                                            max_model_len=max_model_len,
                                            max_new_token=max_new_token,
                                        ),
                                    ),
                                }
                            ]
                        )
                    )
                responses: list[Response] = generator.generate(
                    generation_config=gen_cfg, conversations=conversations
                )

            # 出力を保存
            with pred_filepath.open("a", encoding="utf-8") as f:
                for sample, resp, conv in zip(
                    batch, responses, conversations, strict=True
                ):
                    json.dump(
                        {
                            "id": sample["sample_id"],
                            "difficulty": sample["difficulty"],
                            "length": sample["length"],
                            "token_count": sample["tokens"],
                            "answer": sample["answer"],
                            "input_prompt": conv.messages[-1].content,
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
