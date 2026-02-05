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
        **kwargs: Any,
    ) -> float:
        """
        Prefix 制約付き文字列類似度（Prefix-Match Similarity）。

        予測文字列 `pred_full`（= output.output）が、事前に指定されたランダム文字列
        `prefix`（= output.random_string_to_prepend）を先頭に正しく保持していることを
        必須条件とした上で、prefix 除去後の文字列と正解文字列との類似度を評価する。

        評価手順:
          1. `pred_full` が `prefix` で始まらない場合は 0.0 を返す。
          2. `prefix` を `pred_full` および正解文字列 `ref_full`（= output.answer）から除去し、
             残りの文字列 `pred` と `ref` を取得する。
          3. `pred` と `ref` の文字列類似度を `difflib.SequenceMatcher` により算出し、
             その ratio（[0.0, 1.0]）をスコアとして返す。

        スコア範囲:
          - [0.0, 1.0]
        """
        if not output.output or output.output == default_error_message:
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
