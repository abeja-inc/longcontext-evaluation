import json
from collections import defaultdict
from typing import Any, cast

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Response
from tqdm import tqdm

from .._core.evaluate import mean_score_by_group
from .._core.evaluate.table import (
    BaseTable,
    OutputsTable,
)
from .._core.predict import truncate_text
from .._core.runner import BaseBenchmarkRunner
from ..config import SubtaskConfig
from .evaluate.metrics import OpenAIMRCRMetrics
from .evaluate.table import (
    OpenAIMRCRLeaderBoardTableRow,
    OpenAIMRCROutputsTableRow,
)
from .predict.data import OpenAIMRCRInput, OpenAIMRCROutput
from .settings import OpenAIMRCRSettings


class OpenAIMRCRRunner(
    BaseBenchmarkRunner[
        OpenAIMRCRSettings,
        OpenAIMRCROutput,
        OpenAIMRCROutputsTableRow,
        OpenAIMRCRLeaderBoardTableRow,
    ]
):
    def _build_metrics(
        self,
    ) -> OpenAIMRCRMetrics:
        return OpenAIMRCRMetrics(logger=self.logger)

    @property
    def settings_model(self) -> type[OpenAIMRCRSettings]:
        return OpenAIMRCRSettings

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: OpenAIMRCRSettings,
        batchsize: int,
    ) -> list[OpenAIMRCROutput]:
        if config.inference_mode == "completion":
            raise NotImplementedError("Completion mode is not supported for OpenAIMRCR")

        # Load dataset
        dataset_filepath = config.dataset_filepath
        output_dilepath = config.output_filepath

        all_data: list[OpenAIMRCRInput] = []
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
                    all_data.append(OpenAIMRCRInput.model_validate(record))

        # Skip processed samples
        processed_ids: set[str | int] = set()
        outputs: list[OpenAIMRCROutput] = []
        if output_dilepath.exists():
            with output_dilepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            output = OpenAIMRCROutput(**record)
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

            conversations: list[Conversation] = []
            for sample in batch:
                self.logger.debug("Make input prompt")
                conversation = Conversation.model_validate(
                    {"messages": json.loads(sample.prompt)}
                )

                if settings.use_truncate:
                    self.logger.debug("Truncate input prompt")
                    truncated_conversation = cast(
                        Conversation,
                        truncate_text(
                            text=conversation,
                            max_context_length=generator.max_context_length,
                            max_output_tokens=generator.max_output_tokens,
                            token_counter=generator.token_counter,
                            truncate_type=settings.truncate_type,
                            buffer_tokens=settings.truncate_buffer_tokens,
                            chat_template_kwargs=generation_kwargs.get(
                                "chat_template_kwargs"
                            ),
                        ),
                    )
                else:
                    truncated_conversation = conversation
                conversations.append(truncated_conversation)
            input_prompts += [conv.to_string for conv in conversations]
            self.logger.debug("Inference started")
            responses: list[Response] = generator.chat(
                conversations=conversations, **generation_kwargs
            )
            new_responses += responses

        # Format
        for input, prompt, response in zip(
            filtered_data, input_prompts, new_responses, strict=True
        ):
            outputs.append(
                OpenAIMRCROutput(
                    id=input.id,
                    input=prompt,
                    context_length=input.tokens,
                    output=response.outputs[0].content.strip(),
                    output_reasoning=response.outputs[0].reasoning_content.strip()
                    if response.outputs[0].reasoning_content
                    else None,
                    answer=input.answer,
                    random_string_to_prepend=input.random_string_to_prepend,
                    n_needles=input.n_needles,
                    desired_msg_index=input.desired_msg_index,
                    total_messages=input.total_messages,
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
        settings: OpenAIMRCRSettings,
        output: OpenAIMRCROutput,
        **kwargs: Any,
    ) -> OpenAIMRCROutputsTableRow:
        score = self.metrics.eval(
            output=output, config=config, settings=settings, **kwargs
        )
        return OpenAIMRCROutputsTableRow(
            model_name=model_name,
            task=task,
            subtask=config.name,
            language=config.language,
            score=score,
            **output.model_dump(),
        )

    def _to_outputs_table(
        self, name: str, rows: list[OpenAIMRCROutputsTableRow]
    ) -> OutputsTable[OpenAIMRCROutputsTableRow]:
        return OutputsTable(
            name="openai-mrcr_outputs_table",
            rows=rows,
        )

    def _make_leaderboard_table(
        self, outputs: list[OpenAIMRCROutputsTableRow]
    ) -> BaseTable[OpenAIMRCRLeaderBoardTableRow]:
        leaderboard_by_model: defaultdict[str, dict[str, float]] = defaultdict(dict)

        overall = mean_score_by_group(rows=outputs, group_by=None)
        for model, group_scores in overall.items():
            leaderboard_by_model[model].update(group_scores)

        return BaseTable(
            name="openai-mrcr_leaderboard_table",
            rows=[
                OpenAIMRCRLeaderBoardTableRow(
                    model_name=model, overall=groups["overall"]
                )
                for model, groups in sorted(leaderboard_by_model.items())
            ],
        )

    def _make_additional_tables(
        self, outputs: list[OpenAIMRCROutputsTableRow]
    ) -> list[BaseTable[Any]]:
        return []
