from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml
from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Message
from transformers import AutoTokenizer

from .._base_benchmark.core import read_jsonl
from .._base_benchmark.interfaces import Batch, PredictJob
from ..utils import filter_names


@dataclass(frozen=True)
class LongBenchPromptTemplates:
    templates: dict[str, str]


def load_prompt_templates(tasks_yml: Path) -> LongBenchPromptTemplates:
    with tasks_yml.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    prompt_dir = Path(raw["prompt"]["dirpath"]).expanduser()
    mapping: dict[str, str] = raw["prompt"]["prompts"]

    templates: dict[str, str] = {}
    for prompt_key, prompt_file_name in mapping.items():
        prompt_file = prompt_dir / prompt_file_name
        templates[prompt_key] = prompt_file.read_text(encoding="utf-8")

    return LongBenchPromptTemplates(templates=templates)


def _build_prompt_from_template(
    prompt_templates: dict[str, str],
    sample: dict[str, Any],
    *,
    rag_topn: int = 0,
    cot: bool = False,
    no_context: bool = False,
) -> tuple[str, str]:
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

    q = sample["question"]
    c_a = sample["choice_A"]
    c_b = sample["choice_B"]
    c_c = sample["choice_C"]
    c_d = sample["choice_D"]

    filled = (
        template.replace("$DOC$", doc.strip())
        .replace("$Q$", q.strip())
        .replace("$C_A$", c_a.strip())
        .replace("$C_B$", c_b.strip())
        .replace("$C_C$", c_c.strip())
        .replace("$C_D$", c_d.strip())
    )

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


def _truncate(
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
    return content


@dataclass
class LongBenchPredictJob(PredictJob):
    name: str
    dataset_path: Path
    pred_path: Path
    prompt_templates: LongBenchPromptTemplates
    tokenizer: AutoTokenizer
    max_model_len: int
    max_new_tokens: int
    rag_topn: int = 0
    cot: bool = False
    no_context: bool = False

    def load_processed_ids(self) -> set[str]:
        processed: set[str] = set()
        if not self.pred_path.exists():
            return processed

        with self.pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in record:
                    processed.add(str(record["id"]))
        return processed

    def iter_batches(self, batch_size: int) -> Iterable[Batch]:
        samples = read_jsonl(self.dataset_path)
        processed = self.load_processed_ids()
        remaining = [
            sample
            for sample in samples
            if str(sample.get("sample_id")) not in processed
        ]
        for start in range(0, len(remaining), batch_size):
            yield Batch(samples=remaining[start : start + batch_size])

    def run_batch(
        self,
        *,
        generator: BaseGenerator,
        generate_kwargs: dict[str, Any],
        batch: Batch,
    ) -> list[dict[str, Any]]:
        prompt_templates = self.prompt_templates.templates

        conversations: list[Conversation] = []
        answer_templates: list[str] = []
        for sample in batch.samples:
            user_prompt, answer_template = _build_prompt_from_template(
                prompt_templates=prompt_templates,
                sample=sample,
                rag_topn=self.rag_topn,
                cot=self.cot,
                no_context=self.no_context,
            )
            truncated_user_prompt = _truncate(
                content=user_prompt,
                tokenizer=self.tokenizer,
                max_model_len=self.max_model_len,
                max_new_token=self.max_new_tokens,
            )
            conversations.append(
                Conversation(
                    messages=[Message(role="user", content=truncated_user_prompt)]
                )
            )
            answer_templates.append(answer_template)

        responses = _call_chat(
            generator=generator,
            conversations=conversations,
            generate_kwargs=generate_kwargs,
        )

        if self.cot:
            followup_conversations: list[Conversation] = []
            for response, answer_template in zip(responses, answer_templates, strict=True):
                output = ""
                if response.outputs:
                    output = response.outputs[0].content
                followup_conversations.append(
                    Conversation(
                        messages=[
                            Message(
                                role="assistant",
                                content=answer_template.replace(
                                    "$COT$",
                                    _truncate(
                                        content=output,
                                        tokenizer=self.tokenizer,
                                        max_model_len=self.max_model_len,
                                        max_new_token=self.max_new_tokens,
                                    ),
                                ),
                            )
                        ]
                    )
                )
            responses = _call_chat(
                generator=generator,
                conversations=followup_conversations,
                generate_kwargs=generate_kwargs,
            )

        records: list[dict[str, Any]] = []
        for sample, response, conversation in zip(
            batch.samples, responses, conversations, strict=True
        ):
            prediction = ""
            if response.outputs:
                prediction = response.outputs[0].content
            records.append(
                {
                    "id": sample.get("sample_id"),
                    "difficulty": sample.get("difficulty"),
                    "length": sample.get("length"),
                    "token_count": sample.get("tokens"),
                    "answer": sample.get("answer"),
                    "input_prompt": conversation.messages[-1].content,
                    "prediction": prediction,
                }
            )
        return records


def build_jobs_for_dataset_dir(
    *,
    dataset_dir: Path,
    prediction_dir: Path,
    prompt_templates: LongBenchPromptTemplates,
    tokenizer: AutoTokenizer,
    max_model_len: int,
    max_new_tokens: int,
    rag_topn: int = 0,
    cot: bool = False,
    no_context: bool = False,
    include_datasets: list[str] | None = None,
    exclude_datasets: list[str] | None = None,
) -> list[LongBenchPredictJob]:
    if rag_topn > 0:
        subdir_name = "0shot_rag"
    elif cot:
        subdir_name = "0shot_cot"
    elif no_context:
        subdir_name = "0shot_no_context"
    else:
        subdir_name = "0shot"

    dataset_paths = sorted(dataset_dir.rglob("*_with_token_count.jsonl"))
    dataset_names = [path.name for path in dataset_paths]
    has_filter = bool(include_datasets or exclude_datasets)
    filtered_names = set(
        filter_names(
            dataset_names,
            include=include_datasets,
            exclude=exclude_datasets,
            allow_stem=True,
        )
    )

    jobs: list[LongBenchPredictJob] = []
    for dataset_filepath in dataset_paths:
        if has_filter and dataset_filepath.name not in filtered_names:
            continue
        pred_filepath = (
            prediction_dir / subdir_name / f"{dataset_filepath.stem}.jsonl"
        )
        jobs.append(
            LongBenchPredictJob(
                name=f"{subdir_name}/{dataset_filepath.stem}",
                dataset_path=dataset_filepath,
                pred_path=pred_filepath,
                prompt_templates=prompt_templates,
                tokenizer=tokenizer,
                max_model_len=max_model_len,
                max_new_tokens=max_new_tokens,
                rag_topn=rag_topn,
                cot=cot,
                no_context=no_context,
            )
        )
    return jobs


def _call_chat(
    *,
    generator: BaseGenerator,
    conversations: list[Conversation],
    generate_kwargs: dict[str, Any],
):
    if generate_kwargs.get("sampling_params") is not None:
        return generator.chat(conversations=conversations, **generate_kwargs)

    buffer_tokens = int(generate_kwargs.get("buffer_tokens", 10))
    long_input_filter_kwargs = generate_kwargs.get("long_input_filter_kwargs") or {
        "buffer_tokens": buffer_tokens
    }
    chat_kwargs = generate_kwargs.get("chat_kwargs", {})
    return generator.chat(
        conversations=conversations,
        long_input_filter_kwargs=long_input_filter_kwargs,
        **chat_kwargs,
    )
