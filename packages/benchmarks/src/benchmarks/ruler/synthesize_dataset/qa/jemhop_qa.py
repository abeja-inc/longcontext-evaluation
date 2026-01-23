from typing import Any

from ..data_model import ProcessedQAData, QAPair
from .base import BaseQADatasetGenerator


class JEMHopQAGenerator(BaseQADatasetGenerator):
    def _derivations_to_string(self, derivations: list[dict[str, str]]):
        return "".join(
            [f"{der['0']}の{der['1']}は{der['2']}である。" for der in derivations]
        )

    def _process_data(self, raw_data: list[dict[str, Any]]) -> ProcessedQAData:
        """
         JEMHopQA (Japanese Explainable MultiHop Question-Answering)

        Q&A のペアのリスト形式で、答えを導くための推論過程が記録されたデータになっています。一つのペアデータに含まれる情報は以下の通りです。
        ```
        - qid: "2138f0638f363e75593d09df560db76c" (質問のID)
        - type: "comparison" (質問のタイプ。「比較」や「構成」など)
        - question: "『ダンガンロンパ 希望の学園と絶望の高校生』と『ファイナルファンタジーXIII』、発売日が早いのはどちらでしょう？" (質問文)
        - answers: ["ファイナルファンタジーXIII"] (最終的な答えのリスト)
        - derivations: (答えを導くための推論ステップのリスト)
            - 推論ステップ (主語・属性・値のセット)
                - "0": "ダンガンロンパ 希望の学園と絶望の高校生" (主語エンティティ)
                - "1": "発売日" (主語と目的語のエンティティ間の関係)
                - "2": ["2010年11月25日"] (目的語エンティティ)
        - page_ids: ["543287", "2236928"] (参照元ページのIDリスト)
        - time_dependent: false (答えが時間によって変化するかのフラグ)
        ```
        """
        contexts = [
            self._derivations_to_string(derivations=pair["derivations"])
            for pair in raw_data
        ]
        contexts = sorted(list(set(contexts)))
        idx_contexts = {c: idx for idx, c in enumerate(contexts)}

        qas: list[QAPair] = []
        for _pair in raw_data:
            qas.append(
                QAPair(
                    question=_pair["question"],
                    answers=_pair["answers"],
                    context_indices=[
                        idx_contexts[
                            self._derivations_to_string(
                                derivations=_pair["derivations"]
                            )
                        ]
                    ],
                    more_context_indices=None,
                )
            )

        return ProcessedQAData(contexts=contexts, qas=qas)
