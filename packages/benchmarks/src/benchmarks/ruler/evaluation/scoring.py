from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from ..._base_benchmark.scoring import BaseRecordScorer
from ..data_model import Score


def match_pattern(pred: str, ref: str) -> bool:
    return ref.lower() in pred.lower()


class BaseStringMatcher(ABC):
    def compute(self, preds: list[str], refs: list[list[str]]) -> float:
        if len(preds) != len(refs):
            raise ValueError("preds and refs must be same length")

        total_score = 0.0
        for pred, ref_list in zip(preds, refs, strict=True):
            total_score += self._score(pred, ref_list)

        avg_score = (total_score / len(preds)) * 100
        return round(avg_score, 2)

    @abstractmethod
    def _score(self, pred: str, ref_list: list[str]) -> float:
        raise NotImplementedError


class PartStringMatcher(BaseStringMatcher):
    def _score(self, pred: str, ref_list: list[str]) -> float:
        return max(
            1.0 if match_pattern(pred=pred, ref=ref) else 0.0 for ref in ref_list
        )


class AllStringMatcher(BaseStringMatcher):
    def _score(self, pred: str, ref_list: list[str]) -> float:
        return sum(
            1.0 if match_pattern(pred=pred, ref=ref) else 0.0 for ref in ref_list
        ) / len(ref_list)


class RulerScorer(BaseRecordScorer[list[Score]]):
    name = "ruler"

    def __init__(self, metric: str):
        if metric == "part":
            self._matcher: BaseStringMatcher = PartStringMatcher()
        elif metric == "all":
            self._matcher = AllStringMatcher()
        else:
            raise ValueError(
                f"Unsupported metric: '{metric}' for task. Use 'part' or 'all'."
            )

    def score_records(self, records: list[dict[str, Any]]) -> list[Score]:
        preds: list[str] = [record.get("prediction", "") for record in records]
        refs: list[list[str]] = [record.get("answer", []) for record in records]
        context_lengths: list[int] = [
            record.get("target_context_length", -1) for record in records
        ]
        return self.score(preds=preds, refs=refs, context_lengths=context_lengths)

    def score(
        self, *, preds: list[str], refs: list[list[str]], context_lengths: list[int]
    ) -> list[Score]:
        grouped: dict[int, list[tuple[str, list[str]]]] = defaultdict(list)
        for pred, ref, ctx_len in zip(preds, refs, context_lengths, strict=False):
            grouped[ctx_len].append((pred, ref))

        score_by_context: list[Score] = []
        for ctx_len, pairs in grouped.items():
            grouped_preds, grouped_refs = zip(*pairs, strict=False)
            score = self._matcher.compute(
                preds=list(grouped_preds), refs=list(grouped_refs)
            )
            score_by_context.append(Score(score=score, context_length=ctx_len))

        return score_by_context
