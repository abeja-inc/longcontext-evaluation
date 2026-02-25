from dataclasses import dataclass

from ..._core.evaluate.table import BaseTableRow, OutputsTableRow


@dataclass(frozen=True)
class NemotronPersonaQAOutputsTableRow(OutputsTableRow):
    num_personas: int


@dataclass(frozen=True)
class NemotronPersonaQALeaderBoardTableRow(BaseTableRow):
    overall: float
