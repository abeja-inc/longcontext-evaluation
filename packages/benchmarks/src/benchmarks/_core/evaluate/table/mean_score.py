from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .table import GroupBy, MeanScoreTableRow, OutputsTableRow


@dataclass(frozen=True)
class Bin:
    upper: int
    label: str


def context_bin_label(
    context_length: int, bins: list[Bin], over_label: str = "over"
) -> str:
    for bin in sorted(bins, key=lambda b: b.upper):
        if context_length <= bin.upper:
            return bin.label
    return f"{over_label}_{max(b.upper for b in bins)}"


def _get_group_value(row: OutputsTableRow, group_by: GroupBy) -> str:
    if group_by == "subtask":
        return row.subtask
    if group_by == "task":
        return row.task
    if group_by == "language":
        return row.language
    else:
        raise ValueError(f"Invalid group_by: {group_by}")


def mean_score_by_group_and_context_bin(
    rows: Iterable[OutputsTableRow],
    *,
    group_by: GroupBy,
    context_bins: list[Bin],
    over_label: str = "over",
) -> list[MeanScoreTableRow]:
    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for r in rows:
        group_val = _get_group_value(row=r, group_by=group_by)
        ctx_bin = context_bin_label(
            context_length=r.context_length, bins=context_bins, over_label=over_label
        )

        key = (r.model_name, f"{group_val} | {ctx_bin}")
        sums[key] += float(r.score)
        counts[key] += 1

    return [
        MeanScoreTableRow(
            model_name=model_name,
            group=group,
            mean_score=sums[(model_name, group)] / counts[(model_name, group)],
        )
        for (model_name, group) in sums
    ]
