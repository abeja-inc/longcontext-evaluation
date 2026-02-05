import re
from typing import Any

from ..._core.evaluate.metrics import BaseMetrics
from ...config import SubtaskConfig
from ..predict.data import LongBenchV2Output
from ..settings import LongBenchV2Settings


class LongBenchV2Metrics(BaseMetrics[LongBenchV2Settings, LongBenchV2Output]):
    _metric_registry = {
        "parsed_answer_match": "eval_parsed_answer_match",
    }
    ANSWER_PATTERNS = [
        re.compile(r"The correct answer is \(([A-D])\)"),
        re.compile(r"The correct answer is ([A-D])"),
    ]

    def _extract_answer(self, text: str) -> str | None:
        """
        モデル出力文字列から、事前定義された回答フォーマットに基づいて
        選択肢ラベル（A–D）を抽出する。
        """
        s = str(text).replace("*", "")
        for pat in self.ANSWER_PATTERNS:
            m = pat.search(s)
            if m:
                return m.group(1)
        return None

    def _parsed_answer_match(
        self,
        *,
        output: LongBenchV2Output,
        default_error_message: str,
        compensate_missing: bool = False,
    ) -> float:
        """
        パース済み選択肢一致（Parsed Answer Match）。

        モデル出力 `output.output` から、事前に定義された回答フォーマットに基づいて
        選択肢（A–D）を抽出し、正解ラベル `output.answer` と一致するかを判定する。

        回答抽出:
          - 出力文字列中から、以下のいずれかの正規表現に一致する最初の選択肢を抽出する。
              - "The correct answer is (A|B|C|D)"
              - "The correct answer is A|B|C|D"
          - 装飾目的の `*` 文字は事前に除去される。
        """

        if not output.output or output.output == default_error_message:
            return 0.25 if compensate_missing else 0.0

        parsed_output = self._extract_answer(output.output)
        answer = output.answer if output.answer in ["A", "B", "C", "D"] else None

        if parsed_output is None:
            return 0.25 if compensate_missing else 0.0
        elif answer is None:
            return 0.0
        else:
            return 1.0 if parsed_output == answer else 0.0

    def eval_parsed_answer_match(
        self,
        output: LongBenchV2Output,
        config: SubtaskConfig,
        settings: LongBenchV2Settings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._parsed_answer_match(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )
