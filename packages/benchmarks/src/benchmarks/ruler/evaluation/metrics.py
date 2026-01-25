from abc import ABC, abstractmethod


def match_pattern(pred: str, ref: str) -> bool:
    return ref.lower() in pred.lower()


class BaseStringMatcher(ABC):
    def compute(self, preds: list[str], refs: list[list[str]]) -> float:
        assert len(preds) == len(refs), "preds and refs must be same length"

        total_score = 0.0
        for pred, ref_list in zip(preds, refs, strict=True):
            total_score += self._score(pred, ref_list)

        avg_score = (total_score / len(preds)) * 100
        return round(avg_score, 2)

    @abstractmethod
    def _score(self, pred: str, ref_list: list[str]) -> float:
        pass


class PartStringMatcher(BaseStringMatcher):
    def _score(self, pred: str, ref_list: list[str]) -> float:
        return max([1.0 if match_pattern(pred=pred, ref=ref) else 0.0 for ref in ref_list])


class AllStringMatcher(BaseStringMatcher):
    def _score(self, pred: str, ref_list: list[str]) -> float:
        return sum([1.0 if match_pattern(pred=pred, ref=ref) else 0.0 for ref in ref_list]) / len(
            ref_list
        )
