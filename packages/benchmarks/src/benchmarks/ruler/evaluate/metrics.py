from typing import Any

from rapidfuzz.distance import LCSseq

from ..._core.evaluate.metrics import BaseMetrics
from ...config import SubtaskConfig
from ..predict.data import RULEROutput
from ..settings import RULERSettings


def _match_pattern(pred: str, ref: str) -> bool:
    return ref.lower() in pred.lower()


class RULERMetrics(BaseMetrics[RULERSettings, RULEROutput]):
    _metric_registry = {
        "substr_any": "eval_substr_any",  # metric `part` of RULER
        "substr_coverage": "eval_substr_coverage",  # metric `all` of RULER
        "lcs_f1_max": "eval_lcs_f1_max",  # substitution for `substr_any`
        "lcs_f1_coverage": "eval_lcs_f1_coverage",  # substitution for `substr_coverage`
    }

    def _eval_substr_any(
        self,
        *,
        default_error_message: str,
        output: RULEROutput,
        **kwargs: Any,
    ) -> float:
        """
        部分文字列一致（ANY）。

        予測文字列 `pred`（= output.output）に、正解候補列 `ref_list`（= output.answer）に含まれる
        いずれか 1 つ以上の参照文字列 `ref` が（大文字小文字を無視して）部分文字列として出現する場合 1.0、
        そうでない場合 0.0 を返す。
        """
        if not output.output or output.output == default_error_message:
            return 0.0

        ref_list: list[str] = output.answer
        return max(
            1.0 if _match_pattern(pred=output.output, ref=str(ref)) else 0.0
            for ref in ref_list
        )

    def _eval_substr_coverage(
        self,
        *,
        default_error_message: str,
        output: RULEROutput,
        **kwargs: Any,
    ) -> float:
        """
        部分文字列一致カバレッジ（COVERAGE）。

        予測文字列 `pred`（= output.output）に対し、正解候補列 `ref_list`（= output.answer）の
        各参照文字列 `ref` が（大文字小文字を無視して）部分文字列として出現するかを個別に判定し、
        一致した参照文字列の割合を返す。

        スコア:
          - score = (#hits) / len(ref_list)
          - 取り得る値域は [0.0, 1.0]。
        """
        if not output.output or output.output == default_error_message:
            return 0.0

        ref_list: list[str] = output.answer
        return sum(
            [
                1.0 if _match_pattern(pred=output.output, ref=ref) else 0.0
                for ref in ref_list
            ]
        ) / len(ref_list)

    def _eval_lcs_f1_max(
        self,
        *,
        default_error_message: str,
        output: RULEROutput,
        **kwargs: Any,
    ) -> float:
        """
        文字レベル LCS（Longest Common Subsequence）に基づく F1 の最大値（MAX）。

        予測文字列 `pred`（= output.output）と参照文字列 `ref` の間の LCS 長 `lcs_len` を用い、
        以下で Precision/Recall/F1 を定義する（文字単位）:

            - precision = lcs_len / len(pred)
            - recall    = lcs_len / len(ref)
            - f1        = 2 * precision * recall / (precision + recall)

        正解候補列 `ref_list`（= output.answer）に含まれる各 `ref` について f1 を計算し、
        その最大値をスコアとして返す。
        """
        ref_list: list[str] = output.answer
        if not output.output or output.output == default_error_message:
            return 0.0

        best_f1 = 0.0
        for ref in ref_list:
            ref = ref
            lcs_len = LCSseq.similarity(output.output, ref)
            if lcs_len == 0:
                continue
            precision = lcs_len / len(output.output)
            recall = lcs_len / len(ref)
            f1 = (2 * precision * recall) / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
        return best_f1

    def _eval_lcs_f1_coverage(
        self,
        *,
        default_error_message: str,
        output: RULEROutput,
        **kwargs: Any,
    ) -> float:
        """
        文字レベル LCS に基づく F1 のカバレッジ（COVERAGE）。

        `ref_list` 内の各参照文字列 `ref` について LCS-F1 を計算し、
        その平均値を返す。

        スコア:
            - score = sum_i f1(pred, ref_i) / len(ref_list)
            - 値域は [0.0, 1.0]
        """
        if not output.output or output.output == default_error_message:
            return 0.0

        ref_list: list[str] = output.answer
        if not ref_list:
            return 0.0

        total_f1 = 0.0
        for ref in ref_list:
            ref_norm = str(ref).lower()
            if not ref_norm:
                continue

            lcs_len = LCSseq.similarity(output.output, ref_norm)
            if lcs_len <= 0:
                continue

            pred_len = len(output.output)
            ref_len = len(ref_norm)
            if pred_len == 0 or ref_len == 0:
                continue

            precision = lcs_len / pred_len
            recall = lcs_len / ref_len
            denom = precision + recall
            if denom == 0:
                continue

            f1 = (2 * precision * recall) / denom
            total_f1 += f1

        return total_f1 / len(ref_list)

    def eval_substr_any(
        self,
        output: RULEROutput,
        config: SubtaskConfig,
        settings: RULERSettings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._eval_substr_any(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )

    def eval_substr_coverage(
        self,
        output: RULEROutput,
        config: SubtaskConfig,
        settings: RULERSettings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._eval_substr_coverage(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )

    def eval_lcs_f1_max(
        self,
        output: RULEROutput,
        config: SubtaskConfig,
        settings: RULERSettings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._eval_lcs_f1_max(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )

    def eval_lcs_f1_coverage(
        self,
        output: RULEROutput,
        config: SubtaskConfig,
        settings: RULERSettings,
        default_error_message: str,
        **kwargs: Any,
    ) -> float:
        return self._eval_lcs_f1_coverage(
            output=output,
            default_error_message=default_error_message,
            **settings.metric_kwargs,
            **kwargs,
        )
