from dataclasses import dataclass

from ..._core.evaluate.table import BaseTableRow, OutputsTableRow


@dataclass(frozen=True)
class OpenAIMRCROutputsTableRow(OutputsTableRow):
    random_string_to_prepend: str
    n_needles: int
    desired_msg_index: int
    total_messages: int


@dataclass(frozen=True)
class OpenAIMRCRLeaderBoardTableRow(BaseTableRow):
    overall: float
