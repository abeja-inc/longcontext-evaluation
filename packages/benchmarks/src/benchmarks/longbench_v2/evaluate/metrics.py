import re
from typing import Any

from ..._core.evaluate.metrics import BaseMetrics
from ...config import SubtaskConfig
from ..predict.data import LongBenchV2Output
from ..settings import LongBenchV2Settings


class LongBenchV2Metrics(BaseMetrics[LongBenchV2Settings, LongBenchV2Output]):
    _metric_registry = {
        "exact_match": "eval_exact_match",
    }
    ANSWER_PATTERNS = [
        re.compile(r"The correct answer is \(([A-D])\)"),
        re.compile(r"The correct answer is ([A-D])"),
    ]

    def _extract_answer(self, text: str) -> str | None:
        s = str(text).replace("*", "")
        for pat in self.ANSWER_PATTERNS:
            m = pat.search(s)
            if m:
                return m.group(1)
        return None

    def _exact_match(
        self,
        *,
        output: LongBenchV2Output,
        default_error_message: str,
        compensate_missing: bool = False,
    ) -> float:
        if output.output == default_error_message:
            return 0.25 if compensate_missing else 0.0

        parsed_output = self._extract_answer(output.output)
        answer = output.answer if output.answer in ["A", "B", "C", "D"] else None

        if parsed_output is None:
            return 0.25 if compensate_missing else 0.0
        elif answer is None:
            return 0.0
        else:
            return 1.0 if parsed_output == answer else 0.0

    def eval_exact_match(
        self,
        output: LongBenchV2Output,
        config: SubtaskConfig,
        settings: LongBenchV2Settings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._exact_match(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )
