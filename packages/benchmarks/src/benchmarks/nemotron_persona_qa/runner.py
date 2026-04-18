import json
from collections import defaultdict
from typing import Any

from llm_inference.base import BaseGenerator
from llm_inference.data import Conversation, Response
from tqdm import tqdm

from .._core.evaluate import mean_score_by_group
from .._core.evaluate.table import (
    BaseTable,
    OutputsTable,
)
from .._core.predict import truncate_text
from .._core.predict.truncate import count_conversation_tokens
from .._core.runner import BaseBenchmarkRunner
from ..config import SubtaskConfig
from .evaluate.metrics import NemotronPersonaQAMetrics
from .evaluate.table import (
    NemotronPersonaQALeaderBoardTableRow,
    NemotronPersonaQANumPersonasTableRow,
    NemotronPersonaQAOutputsTableRow,
)
from .predict.data import NemotronPersonaQAInput, NemotronPersonaQAOutput
from .settings import NemotronPersonaQASettings


class NemotronPersonaQARunner(
    BaseBenchmarkRunner[
        NemotronPersonaQASettings,
        NemotronPersonaQAOutput,
        NemotronPersonaQAOutputsTableRow,
        NemotronPersonaQALeaderBoardTableRow,
    ]
):
    def _build_metrics(
        self,
    ) -> NemotronPersonaQAMetrics:
        return NemotronPersonaQAMetrics(logger=self.logger)

    @property
    def settings_model(self) -> type[NemotronPersonaQASettings]:
        return NemotronPersonaQASettings

    def _run_subtask(
        self,
        generator: BaseGenerator,
        generation_kwargs: dict[str, Any],
        config: SubtaskConfig,
        settings: NemotronPersonaQASettings,
        batchsize: int,
    ) -> list[NemotronPersonaQAOutput]:
        if config.inference_mode == "completion":
            raise NotImplementedError(
                "Completion mode is not supported for NemotronPersonaQA"
            )

        # Load dataset
        dataset_filepath = config.dataset_filepath
        output_dilepath = config.output_filepath

        all_data: list[NemotronPersonaQAInput] = []
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
                    all_data.append(NemotronPersonaQAInput.model_validate(record))

        # Skip processed samples
        processed_ids: set[str | int] = set()
        outputs: list[NemotronPersonaQAOutput] = []
        if output_dilepath.exists():
            with output_dilepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            output = NemotronPersonaQAOutput(**record)
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
        context_lengths: list[int] = []
        new_responses: list[Response] = []
        for start in tqdm(
            range(0, total, batchsize),
            desc=f"Processing {dataset_filepath.name}",
            total=int(total / batchsize),
        ):
            batch = filtered_data[start : start + batchsize]

            conversations: list[Conversation] = []
            for sample in batch:
                self.logger.info("Make input prompt")
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
                conversations.append(
                    Conversation.model_validate(
                        {"messages": [{"role": "user", "content": truncated_prompt}]}
                    )
                )

            input_prompts += [conv.to_string for conv in conversations]
            context_lengths += [
                count_conversation_tokens(
                    conv,
                    generator.tokenizer,
                    generator.tokenizer_type,
                    generation_kwargs.get("chat_template_kwargs", {}),
                )
                for conv in conversations
            ]
            self.logger.info("Inference started")
            responses: list[Response] = generator.chat(
                conversations=conversations, **generation_kwargs
            )
            new_responses += responses

        # Format
        for input, prompt, context_length, response in zip(
            filtered_data, input_prompts, context_lengths, new_responses, strict=True
        ):
            outputs.append(
                NemotronPersonaQAOutput(
                    id=input.id,
                    input=prompt,
                    context_length=context_length,
                    output=response.outputs[0].content.strip(),
                    output_reasoning=response.outputs[0].reasoning_content.strip()
                    if response.outputs[0].reasoning_content
                    else None,
                    answer=input.answer,
                    num_personas=input.num_personas,
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
        settings: NemotronPersonaQASettings,
        output: NemotronPersonaQAOutput,
        **kwargs: Any,
    ) -> NemotronPersonaQAOutputsTableRow:
        score = self.metrics.eval(
            output=output, config=config, settings=settings, **kwargs
        )
        return NemotronPersonaQAOutputsTableRow(
            model_name=model_name,
            task=task,
            subtask=config.name,
            language=config.language,
            score=score,
            **output.model_dump(),
        )

    def _to_outputs_table(
        self, name: str, rows: list[NemotronPersonaQAOutputsTableRow]
    ) -> OutputsTable[NemotronPersonaQAOutputsTableRow]:
        return OutputsTable(
            name="nemotron-persona-qa_outputs_table",
            rows=rows,
        )

    def _make_leaderboard_table(
        self, outputs: list[NemotronPersonaQAOutputsTableRow]
    ) -> BaseTable[NemotronPersonaQALeaderBoardTableRow]:
        leaderboard_by_model: defaultdict[str, dict[str, float]] = defaultdict(dict)

        overall = mean_score_by_group(rows=outputs, group_by=None)
        for model, group_scores in overall.items():
            leaderboard_by_model[model].update(group_scores)

        return BaseTable(
            name="nemotron-persona-qa_leaderboard_table",
            rows=[
                NemotronPersonaQALeaderBoardTableRow(
                    model_name=model, overall=groups["overall"]
                )
                for model, groups in sorted(leaderboard_by_model.items())
            ],
        )

    def _make_additional_tables(
        self, outputs: list[NemotronPersonaQAOutputsTableRow]
    ) -> list[BaseTable[Any]]:
        by_num_personas = mean_score_by_group(rows=outputs, group_by="num_personas")
        return [
            BaseTable(
                name="nemotron-persona-qa_accuracy_by_num_personas",
                rows=[
                    NemotronPersonaQANumPersonasTableRow(
                        model_name=model_name,
                        num_personas=int(num_personas),
                        accuracy=score,
                    )
                    for model_name, group_scores in sorted(by_num_personas.items())
                    for num_personas, score in sorted(
                        group_scores.items(), key=lambda item: int(item[0])
                    )
                ],
            )
        ]
