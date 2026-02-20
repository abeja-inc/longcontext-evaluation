import json
from collections import defaultdict
from string import Template
from typing import Any

from tqdm import tqdm

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Response

from .._core.evaluate import mean_score_by_group, mean_score_by_group_and_context_bin
from .._core.evaluate.table import (
    BaseTable,
    MeanScoreByLengthTable,
    OutputsTable,
)
from .._core.predict import truncate_text
from .._core.runner import BaseBenchmarkRunner
from ..config import SubtaskConfig
from .evaluate.metrics import LongBenchV2Metrics
from .evaluate.table import (
    LongBenchV2LeaderBoardTableRow,
    LongBenchV2OutputsTableRow,
)
from .predict import build_input_prompt
from .predict.data import LongBenchV2Input, LongBenchV2Output
from .settings import LongBenchV2Settings


class LongBenchV2Runner(
    BaseBenchmarkRunner[
        LongBenchV2Settings,
        LongBenchV2Output,
        LongBenchV2OutputsTableRow,
        LongBenchV2LeaderBoardTableRow,
    ]
):
    def _build_metrics(
        self,
    ) -> LongBenchV2Metrics:
        return LongBenchV2Metrics(logger=self.logger)

    @property
    def settings_model(self) -> type[LongBenchV2Settings]:
        return LongBenchV2Settings

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: LongBenchV2Settings,
        batchsize: int,
    ) -> list[LongBenchV2Output]:
        if config.inference_mode == "completion":
            raise NotImplementedError(
                "Completion mode is not supported for LongBenchV2"
            )

        # Load dataset
        dataset_filepath = config.dataset_filepath
        output_dilepath = config.output_filepath

        all_data: list[LongBenchV2Input] = []
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
                    all_data.append(LongBenchV2Input.model_validate(record))

        # Skip processed samples
        processed_ids: set[str | int] = set()
        outputs: list[LongBenchV2Output] = []
        if output_dilepath.exists():
            with output_dilepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            output = LongBenchV2Output(**record)
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

        # Select prompt template
        prompt_templates = settings.prompt.load_templates()

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

            cot_templates: list[Template] = []  # For settings.cot == True
            conversations: list[Conversation] = []
            for sample in batch:
                self.logger.debug("Make input prompt")
                user_prompt, cot_template = build_input_prompt(
                    input=sample,
                    prompt_templates=prompt_templates,
                    rag_topn=settings.rag_topn,
                    cot=settings.cot,
                    no_context=settings.no_context,
                )
                if settings.cot:
                    assert cot_template is not None
                    cot_templates.append(cot_template)

                if settings.use_truncate:
                    self.logger.debug("Truncate input prompt")
                    truncated_user_prompt = truncate_text(
                        text=user_prompt,
                        tokenizer=generator.tokenizer,
                        max_context_length=generator.max_context_length,
                        max_output_tokens=generator.max_output_tokens,
                        tokenizer_type=generator.tokenizer_type,
                        truncate_type=settings.truncate_type,
                        buffer_tokens=settings.truncate_buffer_tokens,
                    )
                else:
                    truncated_user_prompt = user_prompt
                conversations.append(
                    Conversation.model_validate(
                        {
                            "messages": [
                                {"role": "user", "content": truncated_user_prompt}
                            ]
                        }
                    )
                )
            input_prompts += [conv.to_string for conv in conversations]
            self.logger.debug("Inference started")
            responses: list[Response] = generator.chat(
                conversations=conversations, **generation_kwargs
            )

            if settings.cot:
                self.logger.info("The Chain-of-thought inference mode is selected")
                conversations: list[Conversation] = []
                for resp, cot_template in zip(responses, cot_templates, strict=True):
                    next_prompt = cot_template.safe_substitute(
                        {"COT": resp.outputs[0].content.strip()}
                    )
                    truncated_next_prompt = truncate_text(
                        text=next_prompt,
                        tokenizer=generator.tokenizer,
                        max_context_length=generator.max_context_length,
                        max_output_tokens=generator.max_output_tokens,
                        tokenizer_type=generator.tokenizer_type,
                        truncate_type=settings.truncate_type,
                    )

                    conversations.append(
                        Conversation.model_validate(
                            {
                                "messages": [
                                    {
                                        "role": "assistant",
                                        "content": cot_template.safe_substitute(
                                            {"COT": truncated_next_prompt}
                                        ),
                                    }
                                ]
                            }
                        )
                    )
                self.logger.info("Chain-of-Thought inference started")
                responses: list[Response] = generator.chat(
                    conversations=conversations, **generation_kwargs
                )
            new_responses += responses

        # Format
        for input, prompt, response in zip(
            filtered_data, input_prompts, new_responses, strict=True
        ):
            outputs.append(
                LongBenchV2Output(
                    id=input.id,
                    input=prompt,
                    context_length=input.tokens,
                    output=response.outputs[0].content.strip(),
                    output_reasoning=response.outputs[0].reasoning_content.strip()
                    if response.outputs[0].reasoning_content
                    else None,
                    answer=input.answer,
                    difficulty=input.difficulty,
                    length=input.length,
                    domain=input.domain,
                    sub_domain=input.sub_domain,
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
        settings: LongBenchV2Settings,
        output: LongBenchV2Output,
        **kwargs: Any,
    ) -> LongBenchV2OutputsTableRow:
        score = self.metrics.eval(
            output=output, config=config, settings=settings, **kwargs
        )
        return LongBenchV2OutputsTableRow(
            model_name=model_name,
            task=task,
            subtask=config.name,
            language=config.language,
            score=score,
            **output.model_dump(),
        )

    def _to_outputs_table(
        self, name: str, rows: list[LongBenchV2OutputsTableRow]
    ) -> OutputsTable[LongBenchV2OutputsTableRow]:
        return OutputsTable(
            name="longbenchv2_outputs_table",
            rows=rows,
        )

    def _make_leaderboard_table(
        self, outputs: list[LongBenchV2OutputsTableRow]
    ) -> BaseTable[LongBenchV2LeaderBoardTableRow]:
        leaderboard_by_model: defaultdict[str, dict[str, float]] = defaultdict(dict)

        for key in ["difficulty", "length"]:
            grouped = mean_score_by_group(rows=outputs, group_by=key)
            for model, group_scores in grouped.items():
                leaderboard_by_model[model].update(group_scores)

        overall = mean_score_by_group(rows=outputs, group_by=None)
        for model, group_scores in overall.items():
            leaderboard_by_model[model].update(group_scores)

        return BaseTable(
            name="longbenchv2_leaderboard_table",
            rows=[
                LongBenchV2LeaderBoardTableRow(
                    model_name=model,
                    overall=groups["overall"],
                    easy=groups["easy"],
                    hard=groups["hard"],
                    short=groups["short"],
                    medium=groups["medium"],
                    long=groups["long"],
                )
                for model, groups in sorted(leaderboard_by_model.items())
            ],
        )

    def _make_additional_tables(
        self, outputs: list[LongBenchV2OutputsTableRow]
    ) -> list[BaseTable[Any]]:
        tables: list[BaseTable[Any]] = []
        for key in ["difficulty", "domain", "sub_domain"]:
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
