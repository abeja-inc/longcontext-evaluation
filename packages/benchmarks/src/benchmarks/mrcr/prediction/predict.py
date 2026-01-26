from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Message, OutputContent

from ..._base_benchmark.interfaces import Batch, PredictJob


@dataclass
class MRCRPredictJob(PredictJob):
    name: str
    dataset_path: Path
    pred_path: Path

    def load_processed_ids(self) -> set[int]:
        processed: set[int] = set()
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
                    try:
                        processed.add(int(record["id"]))
                    except (TypeError, ValueError):
                        continue
        return processed

    def iter_batches(self, batch_size: int) -> Iterable[Batch]:
        data = pd.read_parquet(self.dataset_path)
        processed_ids = self.load_processed_ids()
        if processed_ids:
            data = data.drop(processed_ids, errors="ignore")

        records = [{"id": idx, **row.to_dict()} for idx, row in data.iterrows()]

        for start in range(0, len(records), batch_size):
            yield Batch(samples=records[start : start + batch_size])

    def run_batch(
        self,
        *,
        generator: BaseGenerator,
        generate_kwargs: dict[str, Any],
        batch: Batch,
    ) -> list[dict[str, Any]]:
        chat_template_kwargs = generate_kwargs.get("chat_template_kwargs", {})
        buffer_tokens = int(generate_kwargs.get("buffer_tokens", 10))

        conversations: list[Conversation] = []
        target_context_lengths: list[int] = []
        for sample in batch.samples:
            messages = [
                Message(role=msg["role"], content=msg["content"])
                for msg in json.loads(sample["prompt"])
            ]
            conversation = Conversation(messages=messages)
            token_count = generator._count_tokens(
                input=conversation, **chat_template_kwargs
            )
            conversations.append(conversation)
            target_context_lengths.append(token_count)

        filtered_conversations, skip_idx = generator._filter_long_inputs(
            inputs=conversations,
            max_context_length=generator.max_context_length,
            max_output_tokens=generator.max_output_tokens,
            buffer_tokens=buffer_tokens,
            **chat_template_kwargs,
        )

        responses = []
        if filtered_conversations:
            responses = _call_chat(
                generator=generator,
                conversations=filtered_conversations,
                generate_kwargs=generate_kwargs,
                buffer_tokens=buffer_tokens,
                chat_template_kwargs=chat_template_kwargs,
            )

        reconstructed: list[OutputContent] = []
        response_index = 0
        for idx in range(len(conversations)):
            if idx in skip_idx:
                reconstructed.append(OutputContent(content=""))
            else:
                response = responses[response_index]
                response_index += 1
                content = ""
                if response.outputs:
                    content = response.outputs[0].content
                reconstructed.append(OutputContent(content=content))

        records: list[dict[str, Any]] = []
        for sample, token_count, output, conversation in zip(
            batch.samples,
            target_context_lengths,
            reconstructed,
            conversations,
            strict=True,
        ):
            records.append(
                {
                    "id": sample["id"],
                    "target_context_length": token_count,
                    "random_string_to_prepend": sample.get("random_string_to_prepend"),
                    "answer": sample.get("answer"),
                    "prediction": output.content,
                    "input_prompt": conversation.prompt,
                    "output_content": output.content,
                    "reasoning_content": output.reasoning_content,
                    "tags": sample.get("tags"),
                    "language": sample.get("language"),
                }
            )
        return records


def _call_chat(
    *,
    generator: BaseGenerator,
    conversations: list[Conversation],
    generate_kwargs: dict[str, Any],
    buffer_tokens: int,
    chat_template_kwargs: dict[str, Any],
):
    if generate_kwargs.get("sampling_params") is not None:
        return generator.chat(
            conversations=conversations,
            sampling_params=generate_kwargs.get("sampling_params"),
            buffer_tokens=buffer_tokens,
            chat_template_kwargs=chat_template_kwargs,
        )

    long_input_filter_kwargs = generate_kwargs.get("long_input_filter_kwargs") or {
        "buffer_tokens": buffer_tokens
    }
    chat_kwargs = generate_kwargs.get("chat_kwargs", {})
    return generator.chat(
        conversations=conversations,
        long_input_filter_kwargs=long_input_filter_kwargs,
        **chat_kwargs,
    )
