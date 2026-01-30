import json
from collections import defaultdict
from string import Template
from typing import Any

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Response
from pydantic import ValidationError
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
from .evaluate import (
    LongBenchV2LeaderBoardTableRow,
    LongBenchV2OutputsTableRow,
)
from .evaluate.metrics import LongBenchV2Metrics
from .predict import build_input_prompt, load_prompts
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
        # Load dataset
        dataset_filepath = config.dataset_filepath
        output_dilepath = config.output_filepath

        all_data: list[LongBenchV2Input] = []
        with dataset_filepath.open("r") as f:
            for idx, line in enumerate(f):
                try:
                    record = json.loads(line)
                    all_data.append(LongBenchV2Input.model_validate(record))
                except json.JSONDecodeError:
                    self.logger.warning(
                        "[JSONDecodeError]: Skipping invalid data in %s at line %d",
                        dataset_filepath,
                        idx + 1,
                    )
                except ValidationError:
                    self.logger.warning(
                        "[ValidationError]: Skipping invalid data in %s at line %d",
                        dataset_filepath,
                        idx + 1,
                    )

        # Skip processed samples
        processed_ids = set()
        outputs: list[LongBenchV2Output] = []
        if output_dilepath.exists():
            with output_dilepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            outputs.append(LongBenchV2Output(**record))
                            processed_ids.add(record["id"])
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
        prompt_templates = load_prompts(settings.prompt)

        # Prediction
        total = len(filtered_data)
        new_responses = [Response]
        for start in tqdm(
            range(0, total, batchsize),
            desc=f"Processing {dataset_filepath.name}",
            total=int(total / batchsize),
        ):
            batch = filtered_data[start : start + batchsize]

            if settings.cot:
                cot_templates: list[Template] = []

            conversations: list[Conversation] = []
            for sample in batch:
                user_prompt, cot_template = build_input_prompt(
                    input=sample,
                    prompt_templates=prompt_templates,
                    rag_topn=settings.rag_topn,
                    cot=settings.cot,
                    no_context=settings.no_context,
                )
                if settings.cot:
                    cot_templates.append(cot_template)

                truncated_user_prompt = truncate_text(
                    text=user_prompt,
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
                                {"role": "user", "content": truncated_user_prompt}
                            ]
                        }
                    )
                )
            responses: list[Response] = generator.chat(
                conversations=conversations, **generation_kwargs
            )

            if settings.cot:
                conversations: list[Conversation] = []
                for resp, cot_template in zip(responses, cot_templates, strict=True):
                    next_user_prompt = cot_template.safe_replace(
                        {"COT": resp.outputs[0].content.strip()}
                    )
                    truncated_next_user_prompt = truncate_text(
                        text=next_user_prompt,
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
                                        "content": cot_template.safe_replace(
                                            {"COT": truncated_next_user_prompt}
                                        ),
                                    }
                                ]
                            }
                        )
                    )
                responses: list[Response] = generator.chat(
                    conversations=conversations, **generation_kwargs
                )
            new_responses += responses

        # Format
        for input, response in zip(filtered_data, new_responses, strict=True):
            outputs.append(
                LongBenchV2Output(
                    id=input.id,
                    input=str(input.prompt),
                    context_length=input.tokens,
                    output=response.outputs[0].content.strip(),
                    output_reasoning=response.outputs[0].reasoning_content.strip()
                    if response.outputs[0].reasoning_content
                    else None,
                    answer=input.asnwer,
                    difficulty=input.difficulty,
                    length=input.length,
                    domain=input.domain,
                    sub_domain=input.sub_domain,
                )
            )

        # Overwrite
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
            context_length=output.context_length,
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
    ) -> LongBenchV2LeaderBoardTable:
        leaderboard_dict: dict[str, float] = defaultdict(float)
        for key in ["difficulty", "length"]:
            leaderboard_dict.update(mean_score_by_group(rows=outputs, group_by=key))
        leaderboard_dict.update(mean_score_by_group(rows=outputs, group_by=None))
        return LongBenchV2LeaderBoardTable(
            name="longbenchv2_leaderboard_table",
            rows=[
                LongBenchV2LeaderBoardTableRow(
                    model_name=outputs[0].model_name,
                    overall=leaderboard_dict["overall"],
                    easy=leaderboard_dict["easy"],
                    hard=leaderboard_dict["hard"],
                    short=leaderboard_dict["short"],
                    medium=leaderboard_dict["medium"],
                    long=leaderboard_dict["long"],
                )
            ],
        )

    def _make_additional_tables(
        self, outputs: list[LongBenchV2OutputsTableRow]
    ) -> list[BaseTable[Any]]:
        tables: list[MeanScoreByLengthTable] = []
        for key in ["difficulty", "domain", "subdomain"]:
            tables.append(
                MeanScoreByLengthTable(
                    name=f"{key}_mean_score_by_subtask",
                    rows=mean_score_by_group_and_context_bin(
                        rows=outputs,
                        group_by=key,
                        context_bins=self.bins,
                    ),
                ),
            )
        return tables
