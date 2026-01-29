import json

from llm_inference import get_tokenizer
from llm_inference.base import BaseGenerator
from pydantic import ValidationError
from tqdm import tqdm
from llm_inference.data import Conversation, Prompt

from .._core.evaluate.table import OutputsTable
from .._core.predict import truncate_text
from .._core.runner import BaseBenchmarkRunner
from ..config import SubtaskConfig
from .evaluate import (
    LongBenchV2LeaderBoardTable,
    LongBenchV2OutputsTableRow,
)
from .predict import build_input_prompt, load_prompts
from .predict.data import LongBenchV2Input, LongBenchV2Output
from .settings import LongBenchV2Settings


class LongBenchV2Runner(
    BaseBenchmarkRunner[
        LongBenchV2Settings,
        LongBenchV2Output,
        LongBenchV2OutputsTableRow,
        LongBenchV2LeaderBoardTable,
    ]
):
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
        # Get tokenizer
        tokenizer, tokenizer_type = get_tokenizer(generator)

        # Select prompt template
        prompt_templates = load_prompts(settings.prompt)
        if settings.rag_topn > 0:
            prompt_template = prompt_templates["zero_shot_rag"]
        elif settings.cot:
            prompt_template = prompt_templates["zero_shot_cot"]
        elif settings.no_context:
            prompt_template = prompt_templates["zero_shot_no_context"]
        else:
            prompt_template = prompt_templates["zero_shot"]

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
        if output_dilepath.exists():
            with output_dilepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
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

        # Prediction
        total = len(filtered_data)
        for start in tqdm(
            range(0, total, batchsize),
            desc=f"Processing {dataset_filepath.name}",
            total=int(total / batchsize),
        ):
            batch = filtered_data[start : start + batchsize]
            conversations: list[Conversation] = []
            for sample in batch:
                user_prompt, cot_template = build_input_prompt(
                    input=sample,
                    prompt_templates=prompt_templates,
                    rag_topn=settings.rag_topn,
                    cot=settings.cot,
                    no_context=settings.no_context,
                )
                truncated_user_prompt = truncate_text(
                    text=user_prompt,
                    tokenizer=tokenizer,
                    max_context_length: int,
                    max_output_tokens: int,
                    tokenizer_type: Literal['huggingface', 'tiktoken'],
                    truncate_type:
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

    def _to_output_row(
        self, *, task: str, subtask: str, output: LongBenchV2Output
    ) -> LongBenchV2OutputsTableRow: ...

    def _to_outputs_table(
        self, name: str, rows: list[LongBenchV2OutputsTableRow]
    ) -> OutputsTable[LongBenchV2OutputsTableRow]: ...

    def _make_leaderboard_table(
        self, outputs: list[LongBenchV2OutputsTableRow]
    ) -> LongBenchV2LeaderBoardTable: ...
