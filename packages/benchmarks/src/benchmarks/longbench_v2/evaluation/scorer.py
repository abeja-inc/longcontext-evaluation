import re
from typing import Any

from ..._scoring import BaseRecordScorer, Score


ANSWER_PATTERNS = [
    re.compile(r"The correct answer is \(([A-D])\)"),
    re.compile(r"The correct answer is ([A-D])"),
]


def extract_answer(text: Any) -> str | None:
    normalized = str(text).replace("*", "")
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group(1)
    return None


def to_percent(sum_acc: float, n: int) -> float:
    return round(100.0 * sum_acc / n, 1) if n > 0 else 0.0


def get_value(d: dict, *keys: str, default=None):
    for key in keys:
        if key in d:
            return d[key]
    return default


class LongBenchScorer(BaseRecordScorer[tuple[dict, list[dict]]]):
    name = "longbench_v2"

    def __init__(self, *, compensate_missing: bool = False):
        self._compensate_missing = compensate_missing

    def score_records(self, records: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
        return evaluate_records(records, compensate_missing=self._compensate_missing)


def evaluate_records(
    recs: list[dict], *, compensate_missing: bool = False
) -> tuple[dict, list[dict]]:
    easy = hard = short = medium = long_count = 0
    easy_acc = hard_acc = short_acc = medium_acc = long_acc = 0.0
    rows: list[dict] = []

    for record in recs:
        answer = get_value(record, "answer", "gold", "label")
        pred_raw = get_value(record, "prediction", "pred", "output")
        pred = extract_answer(pred_raw) if pred_raw is not None else None

        if pred is None:
            acc = 0.25 if compensate_missing else 0.0
        elif answer is None:
            acc = 0.0
        else:
            acc = 1.0 if pred == answer else 0.0

        difficulty = get_value(record, "difficulty", default="unknown")
        length = get_value(record, "length", default="unknown")
        token_count = get_value(record, "token_count", "tokens", default=None)
        sample_id = get_value(record, "id", "sample_id", default=None)

        if difficulty == "easy":
            easy += 1
            easy_acc += acc
        elif difficulty == "hard":
            hard += 1
            hard_acc += acc

        if length == "short":
            short += 1
            short_acc += acc
        elif length == "medium":
            medium += 1
            medium_acc += acc
        elif length == "long":
            long_count += 1
            long_acc += acc

        rows.append(
            {
                "sample_id": sample_id,
                "difficulty": difficulty,
                "length": length,
                "answer": answer,
                "prediction": pred_raw,
                "norm_answer": answer,
                "norm_prediction": pred,
                "acc": acc,
                "token_count": token_count,
            }
        )

    n = len(recs)
    metrics = {
        "overall_n": n,
        "overall_acc": to_percent(easy_acc + hard_acc, n) if n > 0 else 0.0,
        "easy_n": easy,
        "easy_acc": to_percent(easy_acc, easy),
        "hard_n": hard,
        "hard_acc": to_percent(hard_acc, hard),
        "short_n": short,
        "short_acc": to_percent(short_acc, short),
        "medium_n": medium,
        "medium_acc": to_percent(medium_acc, medium),
        "long_n": long_count,
        "long_acc": to_percent(long_acc, long_count),
    }
    return metrics, rows


__all__ = ["LongBenchScorer", "extract_answer"]
