from typing import Any

from ..data_model import ProcessedQAData, QAPair
from .base import BaseQADatasetGenerator


class SQuADGenerator(BaseQADatasetGenerator):
    def _process_data(self, raw_data: dict[str, Any]) -> ProcessedQAData:
        """
        SQuAD (Stanford Question Answering Dataset) 2.0

        データ全体は、一つの大きなJSONオブジェクトで構成されています。
        ```
        {
          "version": "v2.0",
          "data": [ ... ]
        }
        ```
        - version: データセットのバージョンを示します。（例: "v2.0"）
        - data: データセットの本体で、複数の「トピック」を含むリスト（配列）になっています。

        data リストの各要素は、一つのトピック（例えば「Normans（ノルマン人）」や「Southern_California（南カリフォルニア）」）
        ```
        {
          "title": "Normans",
          "paragraphs": [ ... ]
        }
        ```
        - title: そのトピックの主題を表す文字列です。
        - paragraphs: そのトピックに関する複数の「段落」と、それに関連するQ&amp;Aのセットを含むリストです。

        paragraphs リストの各要素は、一つの「文脈（context）」となる文章と、それに対する複数の質問応答（qas）で構成される
        ```
        {
          "context": "The Normans (Norman: Nourmands...) were the people...",
          "qas": [ ... ]
        }
        ```
        - context: 質問の答えが含まれている、背景となる文章（段落）です。
        - qas: この context に基づく「質問と答えのペア」のリストです。

        qas リストの各要素が、一つの質問とその答えに関する詳細情報を持つオブジェクトです。ここがこのデータ構造の核となる部分です。
        A. 答えられる質問の場合
        ```
        {
          "question": "In what country is Normandy located?",
          "id": "56ddde6b9a695914005b9628",
          "answers": [
            {"text": "France", "answer_start": 159},
            {"text": "France", "answer_start": 159},
            ...
          ],
          "is_impossible": false
        }
        ```

        B. 答えられない質問の場合 (SQuAD 2.0の特徴)
        ```
        {
          "plausible_answers": [
            {"text": "Normandy", "answer_start": 137}
          ],
          "question": "What is France a region of?",
          "id": "5ad39d53604f3c001a3fe8d2",
          "answers": [],
          "is_impossible": true
        }
        ```

        """
        contexts = [
            para["context"]
            for topic in raw_data["data"]
            for para in topic["paragraphs"]
        ]
        contexts = sorted(list(set(contexts)))
        idx_contexts = {c: idx for idx, c in enumerate(contexts)}

        qas: list[QAPair] = []
        for _topic in raw_data["data"]:
            more_docs = [idx_contexts[para["context"]] for para in _topic["paragraphs"]]
            for _para in _topic["paragraphs"]:
                for _qas in _para["qas"]:
                    if not bool(_qas["is_impossible"]):
                        qas.append(
                            QAPair(
                                question=_qas["question"],
                                answers=[answer["text"] for answer in _qas["answers"]],
                                context_indices=[idx_contexts[_para["context"]]],
                                more_context_indices=[
                                    idx
                                    for idx in more_docs
                                    if idx != idx_contexts[_para["context"]]
                                ],
                            )
                        )

        return ProcessedQAData(contexts=contexts, qas=qas)
