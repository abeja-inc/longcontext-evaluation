from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from .._base_benchmark.scoring import BaseRecordScorer
from .data_model import Score


class Grader:
    def compute(self, pred: str, ref: str, random_string_to_prepend: str) -> float:
        if pred == "Max context tokens exceeded":
            return 0.0
        if not pred.startswith(random_string_to_prepend):
            return 0.0

        pred = pred.removeprefix(random_string_to_prepend)
        ref = ref.removeprefix(random_string_to_prepend)

        return float(SequenceMatcher(None, pred, ref).ratio())

    def grade(
        self, preds: list[str], refs: list[str], random_string_to_prepends: list[str]
    ) -> float:
        grades = [
            self.compute(pred, ref, random_string_to_prepend)
            for pred, ref, random_string_to_prepend in zip(
                preds, refs, random_string_to_prepends, strict=True
            )
        ]

        return sum(grades) / len(grades)


class MrcrScorer(BaseRecordScorer[list[Score]]):
    name = "mrcr"

    def __init__(self, grader: Grader | None = None):
        self._grader = grader or Grader()

    def score_records(self, records: list[dict[str, Any]]) -> list[Score]:
        preds: list[str] = [record.get("prediction", "") for record in records]
        refs: list[list[str]] = [record.get("answer", []) for record in records]
        random_strings: list[str] = [
            record.get("random_string_to_prepend", "") for record in records
        ]
        context_lengths: list[int] = [
            record.get("target_context_length", -1) for record in records
        ]
        return self.score(
            preds=preds,
            refs=refs,
            random_strings=random_strings,
            context_lengths=context_lengths,
        )

    def score(
        self,
        *,
        preds: list[str],
        refs: list[list[str]],
        random_strings: list[str],
        context_lengths: list[int],
    ) -> list[Score]:
        scores: list[Score] = []
        for index, (pred, ref_list, random_string, ctx_len) in enumerate(
            zip(preds, refs, random_strings, context_lengths, strict=False)
        ):
            ref = ref_list[0] if ref_list else ""
            score = self._grader.compute(pred, ref, random_string)
            scores.append(Score(index=index, score=score, context_length=ctx_len))

        return scores
