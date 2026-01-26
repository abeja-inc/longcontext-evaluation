from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm_inference.base import BaseGenerator
from llm_inference.data import OutputContent, Prompt
from transformers import PreTrainedTokenizerBase

from ..._base_benchmark.interfaces import Batch, PredictJob


@dataclass
class RulerPredictJob(PredictJob):
    name: str
    dataset_path: Path
    pred_path: Path
    tokenizer: PreTrainedTokenizerBase

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
        with self.dataset_path.open("r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]

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
        chat_template_kwargs = generate_kwargs.get("chat_template_kwargs", {})
        enable_thinking = chat_template_kwargs.get("enable_thinking", False)
        buffer_tokens = int(generate_kwargs.get("buffer_tokens", 10))

        conversations = [
            [{"role": "user", "content": sample["content"]["user_prompt"]}]
            for sample in batch.samples
        ]

        prompts = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )
        prompts = [
            prompt + sample["content"]["answer_prefix"]
            for prompt, sample in zip(prompts, batch.samples, strict=True)
        ]

        responses = _call_completion(
            generator=generator,
            prompts=[Prompt(prompt=p) for p in prompts],
            generate_kwargs=generate_kwargs,
            buffer_tokens=buffer_tokens,
        )

        outputs: list[OutputContent] = []
        for response in responses:
            if response.outputs:
                outputs.append(response.outputs[0])
            else:
                outputs.append(OutputContent(content=""))

        records: list[dict[str, Any]] = []
        for sample, output in zip(batch.samples, outputs, strict=True):
            records.append(
                {
                    "id": sample.get("sample_id"),
                    "target_context_length": sample.get("target_context_length"),
                    "answer": sample.get("content", {}).get("outputs"),
                    "prediction": output.content,
                }
            )
        return records


def _call_completion(
    *,
    generator: BaseGenerator,
    prompts: list[Prompt],
    generate_kwargs: dict[str, Any],
    buffer_tokens: int,
):
    if generate_kwargs.get("sampling_params") is not None:
        return generator.completion(
            prompts=prompts,
            sampling_params=generate_kwargs.get("sampling_params"),
            buffer_tokens=buffer_tokens,
        )

    long_input_filter_kwargs = generate_kwargs.get("long_input_filter_kwargs") or {
        "buffer_tokens": buffer_tokens
    }
    completion_kwargs = generate_kwargs.get("completion_kwargs", {})
    return generator.completion(
        prompts=prompts,
        long_input_filter_kwargs=long_input_filter_kwargs,
        **completion_kwargs,
    )
