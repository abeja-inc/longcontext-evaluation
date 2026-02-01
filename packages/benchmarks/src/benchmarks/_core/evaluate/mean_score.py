from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .table import MeanScoreByLengthTableRow, OutputsTableRow, OutputsTableRowType


@dataclass(frozen=True)
class Bin:
    upper: int
    label: str


def _context_bin_label(
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
) -> dict[str, dict[str, float]]:
    sums: defaultdict[tuple[str, str], float] = defaultdict(float)
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        if group_by is None:
            key = (row.model_name, "overall")
        else:
            key = (row.model_name, _get_group_value(row=row, group_by=group_by))

        sums[key] += float(row.score)
        counts[key] += 1

    out: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for (model, group), total in sums.items():
        out[model][group] = total / counts[(model, group)]
    return dict(out)


def mean_score_by_group_and_context_bin(
    rows: Iterable[OutputsTableRowType],
    *,
    group_by: str,
    context_bins: list[Bin],
    over_label: str = "over",
) -> list[MeanScoreByLengthTableRow]:
    sums: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)

    for row in rows:
        group_val = _get_group_value(row=row, group_by=group_by)
        ctx_bin = _context_bin_label(
            context_length=row.context_length, bins=context_bins, over_label=over_label
        )

        key = (row.model_name, group_val, ctx_bin)
        sums[key] += float(row.score)
        counts[key] += 1

    return [
        MeanScoreByLengthTableRow(
            model_name=model,
            group=group,
            context_length=context_length,
            mean_score=sums[(model, group, context_length)]
            / counts[(model, group, context_length)],
        )
        for (model, group, context_length) in sorted(sums)
    ]
