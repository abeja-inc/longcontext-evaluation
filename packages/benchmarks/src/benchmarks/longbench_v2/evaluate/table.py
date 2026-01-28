from dataclasses import dataclass

from ..._core.evaluate import BaseTable, BaseTableRow, OutputsTable, OutputsTableRow


@dataclass(frozen=True)
class LongBenchV2OutputsTableRow(OutputsTableRow):
    # 必要なフィールドを追加
    pass


@dataclass(frozen=True)
class LongBenchV2OutputsTable(OutputsTable[LongBenchV2OutputsTableRow]):
    # 必要なフィールドを追加
    pass


@dataclass(frozen=True)
class LongBenchV2LeaderBoardTableRow(BaseTableRow):
    # 必要なフィールドを追加
    pass


@dataclass(frozen=True)
class LongBenchV2LeaderBoardTable(BaseTable[LongBenchV2LeaderBoardTableRow]):
    # 必要なフィールドを追加
    pass
