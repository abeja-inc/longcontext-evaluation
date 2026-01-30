from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .table import MeanScoreByLengthTableRow, OutputsTableRow, OutputsTableRowType


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


def _get_group_value(row: OutputsTableRow, group_by: str) -> str:
    try:
        value = getattr(row, group_by)
    except AttributeError:
        raise ValueError(f"Invalid group_by: {group_by}")
    return str(value)


def mean_score_by_group(
    rows: Iterable[OutputsTableRowType],
    *,
    group_by: str | None = None,
) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for row in rows:
        if group_by is None:
            key = "overall"
        else:
            key = _get_group_value(row=row, group_by=group_by)

        sums[key] += float(row.score)
        counts[key] += 1

    return {key: sums[key] / counts[key] for key in sums}


def mean_score_by_group_and_context_bin(
    rows: Iterable[OutputsTableRowType],
    *,
    group_by: str,
    context_bins: list[Bin],
    over_label: str = "over",
) -> list[MeanScoreByLengthTableRow]:
    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        group_val = _get_group_value(row=row, group_by=group_by)
        ctx_bin = context_bin_label(
            context_length=row.context_length, bins=context_bins, over_label=over_label
        )

        key = (row.model_name, f"{group_val} | {ctx_bin}")
        sums[key] += float(row.score)
        counts[key] += 1

    return [
        MeanScoreByLengthTableRow(
            model_name=model_name,
            group=group,
            mean_score=sums[(model_name, group)] / counts[(model_name, group)],
        )
        for (model_name, group) in sums
    ]
