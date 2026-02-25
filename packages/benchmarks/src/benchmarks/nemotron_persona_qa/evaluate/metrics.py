import re
from typing import Any

from ..._core.evaluate.metrics import BaseMetrics
from ...config import SubtaskConfig
from ..predict.data import NemotronPersonaQAOutput
from ..settings import NemotronPersonaQASettings


class NemotronPersonaQAMetrics(
    BaseMetrics[NemotronPersonaQASettings, NemotronPersonaQAOutput]
):
    _metric_registry = {
        "parsed_answer_match": "eval_parsed_answer_match",
    }

    uuid_pattern = r"([0-9a-fA-F]{32})"
    UUID_PATTERN = re.compile(uuid_pattern)
    ANSWER_PATTERNS = [
        re.compile(
            rf"この特徴に最もよく当てはまる人物の\s*uuid\s*は\s*{uuid_pattern}\s*です"
        ),
        re.compile(rf"uuid\s*は\s*{uuid_pattern}\s*"),
    ]

    def _extract_answer(self, text: str) -> str | None:
        """
        モデル出力文字列から、事前定義された回答フォーマットに基づいて
        uuid を抽出する。
        """
        s = str(text).replace("*", "")
        for pat in self.ANSWER_PATTERNS:
            m = pat.search(s)
            if m:
                return m.group(1)
        return None

    def _parsed_answer_match(
        self, *, output: NemotronPersonaQAOutput, default_error_message: str
    ) -> float:
        """
        パース済み選択肢一致（Parsed Answer Match）。

        モデル出力 `output.output` から、事前に定義された回答フォーマットに基づいて
        uuid を抽出し、正解 uuid `output.answer` と一致するかを判定する。

        回答抽出:
          - 出力文字列中から、以下のいずれかの正規表現に一致する最初の選択肢を抽出する。
              - "この特徴に最もよく当てはまる人物の uuid は xxxxxxxxxxxxxxxx です"
              - "uuid は xxxxxxxxxxxxxxxx "
        """

        if not output.output or output.output == default_error_message:
            return 0.0

        parsed_output = self._extract_answer(output.output)
        answer = (
            output.answer
            if output.answer and self.UUID_PATTERN.fullmatch(output.answer)
            else None
        )

        if parsed_output is None:
            return 0.0
        elif answer is None:
            return 0.0
        else:
            return 1.0 if parsed_output == answer else 0.0

    def eval_parsed_answer_match(
        self,
        output: NemotronPersonaQAOutput,
        config: SubtaskConfig,
        settings: NemotronPersonaQASettings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._parsed_answer_match(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )
