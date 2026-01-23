import json
import random
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Type

from ..base import BaseDatasetGenerator
from ..config import QASynthesisConfig
from ..data_model import Content, ProcessedQAData, QADatasetSchema, QAPair


class BaseQADatasetGenerator(
    BaseDatasetGenerator[QADatasetSchema, QASynthesisConfig], ABC
):
    """
    Abstract Base Class for QA dataset generators.

    This class implements the common logic for generating QA samples (the "template method")
    and delegates the dataset-specific data processing to subclasses via the
    `_process_data` abstract method.
    """

    @property
    def SCHEMA_CLASS(self) -> Type[QADatasetSchema]:
        return QADatasetSchema

    def __init__(self, config: QASynthesisConfig, logger: Logger) -> None:
        super().__init__(config, logger)
        self.contexts: list[str] = []
        self.qas: list[QAPair] = []

    def _prepare(self, **kwargs: Any) -> None:
        self._load_and_process_qa_data()

    def _load_and_process_qa_data(self) -> None:
        """
        Template method to load a JSON file and process its content.
        """
        self.logger.info("Loading dataset from: %s", self.config.qa_dataset_path)
        with self.config.qa_dataset_path.open(encoding="utf-8") as f:
            if self.config.qa_dataset_path.name.endswith(".jsonl"):
                raw_data = [json.loads(line) for line in f if line.strip()]
            elif self.config.qa_dataset_path.name.endswith(".json"):
                raw_data = json.load(f)
            else:
                raise ValueError(
                    "Unsupported file format: %s", self.config.qa_dataset_path.name
                )

        self.logger.info("Processing raw data...")
        source_data = self._process_data(raw_data)
        self.contexts: list[str] = source_data.contexts
        self.qas: list[QAPair] = source_data.qas
        self.logger.info(
            "Successfully processed %s QAs and %s documents.",
            len(self.qas),
            len(self.contexts),
        )

    @abstractmethod
    def _process_data(self, raw_data: Any) -> ProcessedQAData:
        """
        [Abstract Method] Processes the raw data loaded from a JSON file.
        Subclasses MUST implement this method to handle their specific data format.
        """
        raise NotImplementedError

    def _gen_one_sample(
        self, sample_index: int, num_units: int, **kwargs: Any
    ) -> tuple[Content, dict[str, Any]]:
        sample_index += self.config.pre_samples

        # QAを一つピックアップ
        qa_pair_index = sample_index % len(self.qas)
        qa_pair = self.qas[qa_pair_index]

        # QAの文書インデックスを取得
        gold_doc_indices: list[int] = qa_pair.context_indices
        more_doc_indices: list[int] | None = qa_pair.more_context_indices
        selected_indices = list(gold_doc_indices)

        num_distractors_needed = num_units - len(selected_indices)
        if num_distractors_needed > 0:
            # 同じトピックのパラグラフを優先的に distractor として選択
            potential_distractors = (
                list(more_doc_indices) if more_doc_indices is not None else []
            )
            distractors_from_more = random.sample(
                potential_distractors,
                min(len(potential_distractors), num_distractors_needed),
            )
            selected_indices.extend(distractors_from_more)

            # 残りの distractor を他のトピックから選択
            other_indices = [
                i
                for i in range(len(self.contexts))
                if i not in gold_doc_indices and i not in potential_distractors
            ]
            remaining_needed = num_units - len(selected_indices)
            if remaining_needed > 0:
                selected_indices.extend(
                    random.sample(
                        other_indices, min(remaining_needed, len(other_indices))
                    )
                )
        random.shuffle(selected_indices)

        # コンテキストの数が不足している場合、繰り返し使用する
        if num_units > len(self.contexts):
            repeats = (num_units + len(self.contexts) - 1) // len(self.contexts)
            selected_contexts = (self.contexts * repeats)[:num_units]
            random.shuffle(selected_contexts)
        else:
            selected_contexts = [self.contexts[doc_idx] for doc_idx in selected_indices]

        # 選択された context を箇条書きしたテキストを作成
        context_str = "\n\n".join(
            self.config.document_prompt.format(i=i + 1, document=context)
            for i, context in enumerate(selected_contexts)
        )

        # プロンプトテンプレートに値を代入して入力プロンプトを作成
        user_prompt = self.config.task_prompt_template.format(
            query=qa_pair.question, context=context_str
        )

        # gold_doc_indices のうち最初に出現する文書の位置を特定
        total_docs = len(selected_contexts)
        first_gold_idx = None
        for i, doc in enumerate(selected_contexts):
            for gold_idx in gold_doc_indices:
                if doc == self.contexts[gold_idx]:
                    first_gold_idx = i
                    break
            if first_gold_idx is not None:
                break

        if first_gold_idx is not None and total_docs > 0:
            depth_percent = round((first_gold_idx / total_docs) * 100.0, 1)
        else:
            depth_percent = -1.0  # fallback

        content = Content(
            user_prompt=user_prompt,
            answer_prefix=self.config.answer_prefix_template,
            outputs=qa_pair.answers,
        )
        extra_fields = {
            "target_depth_percent": [depth_percent],
            "question": qa_pair.question,
            "target_context": "\n".join(
                [self.contexts[idx] for idx in qa_pair.context_indices]
            ),
        }
        return content, extra_fields
