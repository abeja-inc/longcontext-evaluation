from difflib import SequenceMatcher
from typing import Any

from ..._core.evaluate.metrics import BaseMetrics
from ...config import SubtaskConfig
from ..predict.data import OpenAIMRCROutput
from ..settings import OpenAIMRCRSettings


class OpenAIMRCRMetrics(BaseMetrics[OpenAIMRCRSettings, OpenAIMRCROutput]):
    _metric_registry = {
        "prefix_match_similarity": "eval_prefix_match_similarity",
    }

    def _prefix_match_similarity(
        self,
        *,
        default_error_message: str,
        output: OpenAIMRCROutput,
    ) -> float:
        if output.output == default_error_message:
            return 0
        if not output.output.startswith(output.random_string_to_prepend):
            return 0
        pred = output.output.removeprefix(output.random_string_to_prepend)
        ref = str(output.answer).removeprefix(output.random_string_to_prepend)
        return float(SequenceMatcher(None, pred, ref).ratio())

    def eval_prefix_match_similarity(
        self,
        output: OpenAIMRCROutput,
        config: SubtaskConfig,
        settings: OpenAIMRCRSettings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._prefix_match_similarity(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )
