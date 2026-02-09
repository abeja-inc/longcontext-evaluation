from dataclasses import dataclass

from ..._core.evaluate.table import BaseTableRow, OutputsTableRow


@dataclass(frozen=True)
class RULEROutputsTableRow(OutputsTableRow):
    needle_depth: list[float]


@dataclass(frozen=True)
class RULERLeaderBoardTableRow(BaseTableRow):
    overall: float
