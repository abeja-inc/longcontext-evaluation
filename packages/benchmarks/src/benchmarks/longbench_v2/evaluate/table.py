from dataclasses import dataclass

from ..._core.evaluate.table import BaseTableRow, OutputsTableRow
from ..predict.data import Difficulty, Domain, Length, SubDomain


@dataclass(frozen=True)
class LongBenchV2OutputsTableRow(OutputsTableRow):
    length: Length
    difficulty: Difficulty
    domain: Domain
    sub_domain: SubDomain


@dataclass(frozen=True)
class LongBenchV2LeaderBoardTableRow(BaseTableRow):
    overall: float
    easy: float
    hard: float
    short: float
    medium: float
    long: float
