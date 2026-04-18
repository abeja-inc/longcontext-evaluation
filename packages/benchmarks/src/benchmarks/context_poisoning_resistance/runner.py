import json
from collections import defaultdict
from pathlib import Path
from zlib import crc32
from typing import Any

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Response
from llm_inference.token_counter import TokenCounter
from tqdm import tqdm

from .._core.evaluate import mean_score_by_group
from .._core.evaluate.table import BaseTable, OutputsTable
from .._core.runner import BaseBenchmarkRunner
from ..config import SubtaskConfig
from .evaluate.metrics import (
    ContextPoisoningResistanceMetrics,
    is_valid_equation_text,
)
from .evaluate.table import (
    ContextPoisoningResistanceConditionComparisonRow,
    ContextPoisoningResistanceLeaderBoardTableRow,
    ContextPoisoningResistanceOutputsTableRow,
    ContextPoisoningResistanceSupportLengthRow,
)
from .predict.data import (
    ContextPoisoningResistanceInput,
    ContextPoisoningResistanceOutput,
)
from .predict.poison import poison_answer_text
from .predict.prompt import build_stage1_conversation, build_stage2_conversation
from .settings import ContextPoisoningResistanceSettings


class ContextPoisoningResistanceRunner(
    BaseBenchmarkRunner[
        ContextPoisoningResistanceSettings,
        ContextPoisoningResistanceOutput,
        ContextPoisoningResistanceOutputsTableRow,
        ContextPoisoningResistanceLeaderBoardTableRow,
    ]
):
    def _build_metrics(self) -> ContextPoisoningResistanceMetrics:
        return ContextPoisoningResistanceMetrics(logger=self.logger)

    @property
    def settings_model(self) -> type[ContextPoisoningResistanceSettings]:
        return ContextPoisoningResistanceSettings

    def _load_dataset(
        self, dataset_filepath: Path
    ) -> list[ContextPoisoningResistanceInput]:
        all_data: list[ContextPoisoningResistanceInput] = []
        with dataset_filepath.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    self.logger.warning(
                        "[JSONDecodeError]: Skipping invalid data in %s at line %d: %s",
                        dataset_filepath,
                        idx + 1,
                        str(e),
                    )
                else:
                    all_data.append(
                        ContextPoisoningResistanceInput.model_validate(record)
                    )
        return all_data

    def _load_existing_outputs(
        self, output_filepath: Path
    ) -> tuple[list[ContextPoisoningResistanceOutput], set[str | int]]:
        outputs: list[ContextPoisoningResistanceOutput] = []
        processed_ids: set[str | int] = set()
        if not output_filepath.exists():
            return outputs, processed_ids
        with output_filepath.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    output = ContextPoisoningResistanceOutput.model_validate(
                        json.loads(line)
                    )
                except json.JSONDecodeError:
                    self.logger.warning("Skipping invalid JSON in %s", output_filepath)
                    continue
                outputs.append(output)
                processed_ids.add(output.id)
        return outputs, processed_ids

    def _stage1_cache_path(
        self,
        output_filepath: Path,
        settings: ContextPoisoningResistanceSettings,
    ) -> Path:
        return output_filepath.parent / settings.stage1_cache_filename

    def _load_stage1_cache(self, path: Path) -> dict[str, dict[str, Any]]:
        cache: dict[str, dict[str, Any]] = {}
        if not path.exists():
            return cache
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                puzzle_id = str(record["id"])
                output = str(record["output"])
                is_correct = bool(record.get("is_correct", False))
                cache[puzzle_id] = {
                    "output": output,
                    "is_correct": is_correct,
                }
        return cache

    def _save_stage1_cache(self, path: Path, cache: dict[str, dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for puzzle_id, row in sorted(cache.items()):
                json.dump(
                    {
                        "id": puzzle_id,
                        "output": row["output"],
                        "is_correct": row["is_correct"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")

    def _load_puzzles(
        self, samples: list[ContextPoisoningResistanceInput]
    ) -> dict[str, list[str]]:
        puzzle_numbers: dict[str, list[str]] = {}
        for sample in samples:
            puzzle_numbers.setdefault(sample.target_id, sample.target_numbers)
            for support_id, numbers in sample.support_number_orders.items():
                puzzle_numbers.setdefault(support_id, numbers)
        return puzzle_numbers

    def _generate_stage1_cache(
        self,
        *,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        batchsize: int,
        settings: ContextPoisoningResistanceSettings,
        samples: list[ContextPoisoningResistanceInput],
        output_filepath: Path,
    ) -> dict[str, dict[str, Any]]:
        cache_path = self._stage1_cache_path(output_filepath, settings)
        cache = self._load_stage1_cache(cache_path)
        puzzles = self._load_puzzles(samples)
        missing_ids = [puzzle_id for puzzle_id in puzzles if puzzle_id not in cache]
        if not missing_ids:
            return cache

        self.logger.info("Generating stage1 cache for %d puzzles", len(missing_ids))
        for start in tqdm(
            range(0, len(missing_ids), batchsize),
            desc="Stage1 context poisoning",
            total=max(1, len(missing_ids) // batchsize),
        ):
            batch_ids = missing_ids[start : start + batchsize]
            conversations = [
                build_stage1_conversation(
                    numbers=puzzles[puzzle_id],
                    settings=settings,
                )
                for puzzle_id in batch_ids
            ]
            responses: list[Response] = generator.chat(
                conversations=conversations, **generation_kwargs
            )
            for puzzle_id, response in zip(batch_ids, responses, strict=True):
                output_text = response.outputs[0].content.strip()
                cache[puzzle_id] = {
                    "output": output_text,
                    "is_correct": is_valid_equation_text(output_text, puzzles[puzzle_id]),
                }

        self._save_stage1_cache(cache_path, cache)
        return cache

    def _build_support_examples(
        self,
        *,
        sample: ContextPoisoningResistanceInput,
        stage1_cache: dict[str, dict[str, Any]],
    ) -> tuple[list[tuple[list[str], str, str]], list[str], list[str]]:
        support_examples: list[tuple[list[str], str, str]] = []
        poisoned_support_outputs: list[str] = []
        skipped_support_ids: list[str] = []
        judge_label = sample.judge_label
        for support_idx, support_id in enumerate(sample.support_ids):
            stage1_row = stage1_cache[support_id]
            if not stage1_row["is_correct"]:
                skipped_support_ids.append(support_id)
                continue

            answer = str(stage1_row["output"])
            if sample.condition != "clean":
                seed_text = f"{sample.id}:{sample.condition}:{support_id}:{support_idx}"
                answer = poison_answer_text(
                    answer,
                    seed=crc32(seed_text.encode("utf-8")),
                )
            poisoned_support_outputs.append(answer)
            support_examples.append(
                (sample.support_number_orders[support_id], answer, judge_label)
            )
        return support_examples, poisoned_support_outputs, skipped_support_ids

    def _count_context_tokens(
        self,
        *,
        generator: BaseGenerator,
        conversation: Conversation,
        generation_kwargs: dict[str, Any],
    ) -> int:
        token_counter = TokenCounter(
            tokenizer_name_or_path=generator.model_name,
            tokenizer_type=generator.tokenizer_type,
        )
        return token_counter.count_tokens(
            conversation,
            **generation_kwargs.get("chat_template_kwargs", {}),
        )

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: ContextPoisoningResistanceSettings,
        batchsize: int,
    ) -> list[ContextPoisoningResistanceOutput]:
        if config.inference_mode == "completion":
            raise NotImplementedError(
                "Completion mode is not supported for ContextPoisoningResistance"
            )

        all_data = self._load_dataset(config.dataset_filepath)
        outputs, processed_ids = self._load_existing_outputs(config.output_filepath)
        filtered_data = [sample for sample in all_data if sample.id not in processed_ids]
        self.logger.info(
            "Skip %s already processed samples. Remaining: %d / %d",
            len(processed_ids),
            len(filtered_data),
            len(all_data),
        )

        stage1_cache = self._generate_stage1_cache(
            generator=generator,
            generation_kwargs=generation_kwargs,
            batchsize=batchsize,
            settings=settings,
            samples=all_data,
            output_filepath=config.output_filepath,
        )

        for start in tqdm(
            range(0, len(filtered_data), batchsize),
            desc=f"Processing {config.dataset_filepath.name}",
            total=max(1, len(filtered_data) // batchsize),
        ):
            batch = filtered_data[start : start + batchsize]
            conversations: list[Conversation] = []
            batch_metadata: list[
                tuple[ContextPoisoningResistanceInput, list[str], list[str]]
            ] = []
            for sample in batch:
                (
                    support_examples,
                    poisoned_support_outputs,
                    skipped_support_ids,
                ) = self._build_support_examples(
                    sample=sample,
                    stage1_cache=stage1_cache,
                )
                conversation = build_stage2_conversation(
                    target_numbers=sample.target_numbers,
                    support_examples=support_examples,
                    target_judge_label=sample.judge_label,
                    settings=settings,
                )
                conversations.append(conversation)
                batch_metadata.append(
                    (sample, poisoned_support_outputs, skipped_support_ids)
                )

            responses: list[Response] = generator.chat(
                conversations=conversations, **generation_kwargs
            )

            for (
                sample,
                poisoned_support_outputs,
                skipped_support_ids,
            ), conversation, response in zip(
                batch_metadata, conversations, responses, strict=True
            ):
                final_output = response.outputs[0].content.strip()
                outputs.append(
                    ContextPoisoningResistanceOutput(
                        id=sample.id,
                        input=conversation.to_string,
                        context_length=self._count_context_tokens(
                            generator=generator,
                            conversation=conversation,
                            generation_kwargs=generation_kwargs,
                        ),
                        output=final_output,
                        output_reasoning=response.outputs[0].reasoning_content.strip()
                        if response.outputs[0].reasoning_content
                        else None,
                        answer="10",
                        target_id=sample.target_id,
                        requested_support_ids=sample.support_ids,
                        support_ids=[
                            support_id
                            for support_id in sample.support_ids
                            if support_id not in skipped_support_ids
                        ],
                        skipped_support_ids=skipped_support_ids,
                        condition=sample.condition,
                        requested_support_length=len(sample.support_ids),
                        support_length=len(sample.support_ids)
                        - len(skipped_support_ids),
                        used_stage1_support_count=len(sample.support_ids)
                        - len(skipped_support_ids),
                        target_numbers=sample.target_numbers,
                        stage1_target_output=str(stage1_cache[sample.target_id]["output"]),
                        stage1_target_is_correct=bool(
                            stage1_cache[sample.target_id]["is_correct"]
                        ),
                        poisoned_support_outputs=poisoned_support_outputs,
                    )
                )

        self.logger.info("Output llm responses to %s", config.output_filepath)
        config.output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with config.output_filepath.open("w", encoding="utf-8") as f:
            for out in outputs:
                json.dump(out.model_dump(), f, ensure_ascii=False)
                f.write("\n")

        return outputs

    def _evaluate_subtask(
        self,
        *,
        model_name: str,
        task: str,
        config: SubtaskConfig,
        settings: ContextPoisoningResistanceSettings,
        output: ContextPoisoningResistanceOutput,
        **kwargs: Any,
    ) -> ContextPoisoningResistanceOutputsTableRow:
        score = self.metrics.eval(
            output=output, config=config, settings=settings, **kwargs
        )
        return ContextPoisoningResistanceOutputsTableRow(
            model_name=model_name,
            task=task,
            subtask=config.name,
            language=config.language,
            score=score,
            final_is_correct=score >= 1.0,
            **output.model_dump(),
        )

    def _to_outputs_table(
        self,
        name: str,
        rows: list[ContextPoisoningResistanceOutputsTableRow],
    ) -> OutputsTable[ContextPoisoningResistanceOutputsTableRow]:
        return OutputsTable(name="context_poisoning_resistance_outputs_table", rows=rows)

    def _make_leaderboard_table(
        self, outputs: list[ContextPoisoningResistanceOutputsTableRow]
    ) -> BaseTable[ContextPoisoningResistanceLeaderBoardTableRow]:
        overall = mean_score_by_group(rows=outputs, group_by=None)
        by_condition = mean_score_by_group(rows=outputs, group_by="condition")

        rows: list[ContextPoisoningResistanceLeaderBoardTableRow] = []
        for model_name, scores in overall.items():
            clean = by_condition.get(model_name, {}).get("clean", 0.0)
            poisoned = by_condition.get(model_name, {}).get("poisoned", 0.0)
            poisoned_marked_incorrect = by_condition.get(model_name, {}).get(
                "poisoned_marked_incorrect", 0.0
            )
            rows.append(
                ContextPoisoningResistanceLeaderBoardTableRow(
                    model_name=model_name,
                    overall=scores["overall"],
                    clean=clean,
                    poisoned=poisoned,
                    poisoned_marked_incorrect=poisoned_marked_incorrect,
                    poison_drop=clean - poisoned,
                )
            )
        return BaseTable(
            name="context_poisoning_resistance_leaderboard_table",
            rows=rows,
        )

    def _make_additional_tables(
        self, outputs: list[ContextPoisoningResistanceOutputsTableRow]
    ) -> list[BaseTable[Any]]:
        grouped: defaultdict[
            tuple[str, str, int], dict[str, ContextPoisoningResistanceOutputsTableRow]
        ] = defaultdict(dict)
        for row in outputs:
            key = (row.model_name, row.target_id, row.requested_support_length)
            grouped[key][row.condition] = row

        comparison_rows: list[ContextPoisoningResistanceConditionComparisonRow] = []
        for (model_name, target_id, requested_support_length), rows_by_condition in sorted(
            grouped.items()
        ):
            clean_row = rows_by_condition.get("clean")
            poisoned_row = rows_by_condition.get("poisoned")
            marked_row = rows_by_condition.get("poisoned_marked_incorrect")

            clean_score = clean_row.score if clean_row else None
            poisoned_score = poisoned_row.score if poisoned_row else None
            marked_score = marked_row.score if marked_row else None

            poison_delta = (
                None
                if clean_score is None or poisoned_score is None
                else poisoned_score - clean_score
            )
            marked_delta = (
                None if poisoned_score is None or marked_score is None else marked_score - poisoned_score
            )

            comparison_rows.append(
                ContextPoisoningResistanceConditionComparisonRow(
                    model_name=model_name,
                    target_id=target_id,
                    requested_support_length=requested_support_length,
                    actual_support_length=clean_row.support_length
                    if clean_row
                    else poisoned_row.support_length
                    if poisoned_row
                    else marked_row.support_length
                    if marked_row
                    else 0,
                    clean_score=clean_score,
                    poisoned_score=poisoned_score,
                    poisoned_marked_incorrect_score=marked_score,
                    poison_delta=poison_delta,
                    marked_incorrect_delta=marked_delta,
                    changed_by_poison=None
                    if clean_score is None or poisoned_score is None
                    else clean_score != poisoned_score,
                    changed_by_marked_incorrect=None
                    if poisoned_score is None or marked_score is None
                    else poisoned_score != marked_score,
                )
            )

        by_condition_and_support_length: defaultdict[
            tuple[str, str], list[ContextPoisoningResistanceOutputsTableRow]
        ] = defaultdict(list)
        for row in outputs:
            by_condition_and_support_length[(row.model_name, row.condition)].append(row)

        support_length_rows: list[ContextPoisoningResistanceSupportLengthRow] = []
        for (model_name, condition), condition_rows in sorted(
            by_condition_and_support_length.items()
        ):
            grouped_scores = mean_score_by_group(
                rows=condition_rows, group_by="requested_support_length"
            )
            for requested_support_length, accuracy in sorted(
                grouped_scores[model_name].items(), key=lambda item: int(item[0])
            ):
                support_length_rows.append(
                    ContextPoisoningResistanceSupportLengthRow(
                        model_name=model_name,
                        condition=condition,
                        requested_support_length=int(requested_support_length),
                        accuracy=accuracy,
                    )
                )

        return [
            BaseTable(
                name="context_poisoning_resistance_condition_comparison_table",
                rows=comparison_rows,
            ),
            BaseTable(
                name="context_poisoning_resistance_accuracy_by_support_length",
                rows=support_length_rows,
            ),
        ]
