from __future__ import annotations

import pytest

from benchmarks._core.evaluate.mean_score import Bin, mean_score_by_group, mean_score_by_group_and_context_bin
from benchmarks._core.evaluate.table import OutputsTableRow


def _row(model: str, task: str, score: float, context_length: int) -> OutputsTableRow:
    return OutputsTableRow(
        id="1",
        model_name=model,
        task=task,
        subtask="sub",
        language="english",
        context_length=context_length,
        input="in",
        answer="ans",
        score=score,
        output="out",
        output_reasoning=None,
    )


def test_mean_score_group_by_valid_and_invalid() -> None:
    rows = [_row("m1", "task", 1.0, 5), _row("m1", "task", 0.0, 5)]
    grouped = mean_score_by_group(rows, group_by="task")
    assert grouped["m1"]["task"] == 0.5

    single = mean_score_by_group([_row("m2", "task", 0.75, 5)], group_by=None)
    assert single["m2"]["overall"] == 0.75

    with pytest.raises(ValueError, match="Invalid group_by"):
        mean_score_by_group(rows, group_by="missing")


def test_mean_score_context_bins_over_label_and_counts() -> None:
    rows = [_row("m1", "task", 1.0, 5), _row("m1", "task", 0.0, 20)]
    bins = [Bin(upper=10, label="~10")]
    result = mean_score_by_group_and_context_bin(
        rows=rows, group_by="task", context_bins=bins, over_label="over"
    )
    labels = {row.context_length for row in result}
    assert labels == {"~10", "over_10"}
