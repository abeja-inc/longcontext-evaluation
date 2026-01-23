from ..data_model import ProcessedQAData, QAPair
from .base import BaseQADatasetGenerator


class HotpotQAGenerator(BaseQADatasetGenerator):
    def _process_data(self, raw_data: list) -> ProcessedQAData:
        """
        Processes raw data from a HotpotQA-formatted file.

        HotpotQA のデータ構造: 以下の dictionary の list
        {
            "_id": str,                     # サンプルのユニークID
            "answer": str,                  # 答え（自由形式 or yes/no）
            "question": str,                # 質問文（多段推論が必要なもの）
            "supporting_facts": List[List[str | int]],  # 回答の根拠となる文（タイトル + 文インデックス）
            "context": List[List[str | List[str]]],     # Wikipedia記事単位で与えられた文群
            "type": str,                    # 質問の種類（例: "comparison", "bridge"）
            "level": str                    # 難易度（例: "easy", "medium", "hard"）
        }
        """
        contexts = [
            f"{title}\n{''.join(passage)}"
            for sample in raw_data
            for title, passage in sample["context"]
        ]
        contexts = sorted(list(set(contexts)))

        idx_contexts = {c: idx for idx, c in enumerate(contexts)}

        qas: list[QAPair] = []
        for sample in raw_data:
            qas.append(
                QAPair(
                    question=sample["question"],
                    answers=[sample["answer"]],
                    context_indices=[
                        idx_contexts[f"{title}\n{''.join(passage)}"]
                        for title, passage in sample["context"]
                    ],
                    more_context_indices=None,
                )
            )

        return ProcessedQAData(contexts=contexts, qas=qas)
