from ..data_model import ProcessedQAData, QAPair
from .base import BaseQADatasetGenerator


class JSQuADGenerator(BaseQADatasetGenerator):
    def _process_data(self, raw_data: dict) -> ProcessedQAData:
        """
        JSQuAD (Japanese Stanford Question Answering Dataset)

        Q&A のペアのリスト形式のデータになっている。一つのペアデータに含まれる情報は以下の通り。
        ```
        - id: "a10336p0q0"  (各セットを識別するID)
        - title: "梅雨"  (全体のテーマ)
        - context: "梅雨 [SEP] 梅雨（つゆ、ばいう）は、..."  (答えの根拠となる文章)
        - question: "日本で梅雨がないのは北海道とどこか。"  (質問文)
        - answers: (答えの詳細情報が入った、一つ下の階層)
            - answer_start: [25]  (context内で答えが始まる文字位置)
            - text: ["小笠原諸島"]  (答えのテキスト)
        - is_impossible: false  (この文章から回答不可能かを示すフラグ)
        ```
        """
        contexts = [pair["context"] for pair in raw_data]
        contexts = sorted(list(set(contexts)))
        idx_contexts = {c: idx for idx, c in enumerate(contexts)}

        qas: list[QAPair] = []
        for _pair in raw_data:
            if not bool(_pair["is_impossible"]):
                qas.append(
                    QAPair(
                        question=_pair["question"],
                        answers=_pair["answers"]["text"],
                        context_indices=[idx_contexts[_pair["context"]]],
                        more_context_indices=None,
                    )
                )

        return ProcessedQAData(contexts=contexts, qas=qas)
