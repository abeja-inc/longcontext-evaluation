from dataclasses import dataclass

from ..._core.evaluate.table import BaseTableRow, OutputsTableRow


@dataclass(frozen=True)
class NemotronPersonaQAOutputsTableRow(OutputsTableRow):
    num_personas: int


@dataclass(frozen=True)
class NemotronPersonaQALeaderBoardTableRow(BaseTableRow):
    overall: float


@dataclass(frozen=True)
class NemotronPersonaQANumPersonasTableRow(BaseTableRow):
    num_personas: int
    accuracy: float
