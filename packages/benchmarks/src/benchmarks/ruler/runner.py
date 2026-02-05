import json
from collections import defaultdict
from typing import Any

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Prompt, Response
from tqdm import tqdm

from .._core.evaluate import mean_score_by_group, mean_score_by_group_and_context_bin
from .._core.evaluate.table import (
    BaseTable,
    MeanScoreByLengthTable,
    OutputsTable,
)
from .._core.predict import truncate_text
from .._core.runner import BaseBenchmarkRunner
from ..config import SubtaskConfig
from .evaluate.metrics import RULERMetrics
from .evaluate.table import (
    RULERLeaderBoardTableRow,
    RULEROutputsTableRow,
)
from .predict.data import RULERInput, RULEROutput
from .settings import RULERSettings


class RULERRunner(
    BaseBenchmarkRunner[
        RULERSettings,
        RULEROutput,
        RULEROutputsTableRow,
        RULERLeaderBoardTableRow,
    ]
):
    def _build_metrics(
        self,
    ) -> RULERMetrics:
        return RULERMetrics(logger=self.logger)

    @property
    def settings_model(self) -> type[RULERSettings]:
        return RULERSettings

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: RULERSettings,
        batchsize: int,
    ) -> list[RULEROutput]:
        # Load dataset
        dataset_filepath = config.dataset_filepath
        output_dilepath = config.output_filepath

        all_data: list[RULERInput] = []
        with dataset_filepath.open("r") as f:
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
                    all_data.append(RULERInput.model_validate(record))

        # Skip processed samples
        processed_ids: set[str | int] = set()
        outputs: list[RULEROutput] = []
        if output_dilepath.exists():
            with output_dilepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            output = RULEROutput(**record)
                            outputs.append(output)
                            processed_ids.add(output.id)
                        except json.JSONDecodeError:
                            self.logger.warning(
                                f"Skipping invalid JSON in {output_dilepath}"
                            )

        filtered_data = [
            sample for sample in all_data if sample.id not in processed_ids
        ]
        self.logger.info(
            "Skip %s already processed samples. Remaining: %d / %d",
            len(processed_ids),
            len(filtered_data),
            len(all_data),
        )

        # Prediction
        total = len(filtered_data)
        input_prompts: list[str] = []
        new_responses: list[Response] = []
        for start in tqdm(
            range(0, total, batchsize),
            desc=f"Processing {dataset_filepath.name}",
            total=int(total / batchsize),
        ):
            batch = filtered_data[start : start + batchsize]
            truncated_prompts: list[str] = []
            for sample in batch:
                if settings.use_truncate:
                    self.logger.info("Truncate input prompt")
                    truncated_prompt = truncate_text(
                        text=sample.prompt,
                        tokenizer=generator.tokenizer,
                        max_context_length=generator.max_context_length,
                        max_output_tokens=generator.max_output_tokens,
                        tokenizer_type=generator.tokenizer_type,
                        truncate_type=settings.truncate_type,
                        buffer_tokens=settings.truncate_buffer_tokens,
                    )
                else:
                    truncated_prompt = sample.prompt
                truncated_prompts.append(truncated_prompt)

            if (
                config.inference_mode == "chat"
                or generator.tokenizer_type == "tiktoken"
            ):
                self.logger.info("Make input prompt")
                conversations: list[Conversation] = [
                    Conversation.model_validate(
                        {"messages": [{"role": "user", "content": user_prompt}]}
                    )
                    for user_prompt in truncated_prompts
                ]
                self.logger.info("Inference started")
                responses: list[Response] = generator.chat(
                    conversations=conversations, **generation_kwargs
                )
                input_prompts += [conv.to_string for conv in conversations]
            elif (
                config.inference_mode == "completion"
                and generator.tokenizer_type == "huggingface"
            ):
                self.logger.info("Make input prompt")
                prompts_with_template = generator.tokenizer.apply_chat_template(
                    [
                        [{"role": "user", "content": user_prompt}]
                        for user_prompt in truncated_prompts
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                    **generation_kwargs.get("chat_template_kwargs", {}),
                )

                prompts_with_answer_prefix = [
                    prompt + sample.answer_prefix
                    for prompt, sample in zip(prompts_with_template, batch, strict=True)
                ]
                prompts: list[Prompt] = [
                    Prompt.model_validate({"prompt": _prompt})
                    for _prompt in prompts_with_answer_prefix
                ]
                self.logger.info("Inference started")
                responses: list[Response] = generator.completion(
                    prompts=prompts, **generation_kwargs
                )
                input_prompts += [_prompt.to_string for _prompt in prompts]
            else:
                raise ValueError(f"Invalid inference mode: {config.inference_mode}")
            new_responses += responses

        # Format
        for input, prompt, response in zip(
            filtered_data, input_prompts, new_responses, strict=True
        ):
            outputs.append(
                RULEROutput(
                    id=input.id,
                    input=prompt,
                    context_length=input.tokens,
                    output=response.outputs[0].content.strip(),
                    output_reasoning=response.outputs[0].reasoning_content.strip()
                    if response.outputs[0].reasoning_content
                    else None,
                    answer=input.answer,
                    needle_depth=input.needle_depth,
                )
            )

        # Overwrite
        self.logger.info("Output llm responses to %s", config.output_filepath)
        config.output_filepath.parent.mkdir(parents=True, exist_ok=True)
        with config.output_filepath.open("w") as f:
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
        settings: RULERSettings,
        output: RULEROutput,
        **kwargs: Any,
    ) -> RULEROutputsTableRow:
        score = self.metrics.eval(
            output=output, config=config, settings=settings, **kwargs
        )
        return RULEROutputsTableRow(
            model_name=model_name,
            task=task,
            subtask=config.name,
            language=config.language,
            score=score,
            **output.model_dump(),
        )

    def _to_outputs_table(
        self, name: str, rows: list[RULEROutputsTableRow]
    ) -> OutputsTable[RULEROutputsTableRow]:
        return OutputsTable(
            name="ruler_outputs_table",
            rows=rows,
        )

    def _make_leaderboard_table(
        self, outputs: list[RULEROutputsTableRow]
    ) -> BaseTable[RULERLeaderBoardTableRow]:
        leaderboard_by_model: defaultdict[str, dict[str, float]] = defaultdict(dict)

        overall = mean_score_by_group(rows=outputs, group_by=None)
        for model, group_scores in overall.items():
            leaderboard_by_model[model].update(group_scores)

        return BaseTable(
            name="ruler_leaderboard_table",
            rows=[
                RULERLeaderBoardTableRow(model_name=model, overall=groups["overall"])
                for model, groups in sorted(leaderboard_by_model.items())
            ],
        )

    def _make_additional_tables(
        self, outputs: list[RULEROutputsTableRow]
    ) -> list[BaseTable[Any]]:
        tables: list[BaseTable[Any]] = []
        for key in ["needle_depth"]:
            tables.append(
                MeanScoreByLengthTable(
                    name=f"{key}_mean_score_by_context_length",
                    rows=mean_score_by_group_and_context_bin(
                        rows=outputs,
                        group_by=key,
                        context_bins=self.bins,
                    ),
                ),
            )
        return tables
