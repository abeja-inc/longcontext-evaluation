from dataclasses import dataclass

from ..._core.evaluate.table import BaseTableRow, OutputsTableRow


@dataclass(frozen=True)
class RULEROutputsTableRow(OutputsTableRow):
    target_context_length: int
    target_depth_percent: float


@dataclass(frozen=True)
class RULERLeaderBoardTableRow(BaseTableRow):
    overall: float
