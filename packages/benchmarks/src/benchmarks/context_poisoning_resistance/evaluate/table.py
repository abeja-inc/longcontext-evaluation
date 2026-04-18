from dataclasses import dataclass

from ..._core.evaluate.table import BaseTableRow, OutputsTableRow


@dataclass(frozen=True)
class ContextPoisoningResistanceOutputsTableRow(OutputsTableRow):
    target_id: str
    requested_support_ids: list[str]
    support_ids: list[str]
    skipped_support_ids: list[str]
    condition: str
    requested_support_length: int
    support_length: int
    used_stage1_support_count: int
    target_numbers: list[str]
    stage1_target_output: str
    stage1_target_is_correct: bool
    poisoned_support_outputs: list[str]
    final_is_correct: bool


@dataclass(frozen=True)
class ContextPoisoningResistanceLeaderBoardTableRow(BaseTableRow):
    overall: float
    clean: float
    poisoned: float
    poisoned_marked_incorrect: float
    poison_drop: float


@dataclass(frozen=True)
class ContextPoisoningResistanceConditionComparisonRow(BaseTableRow):
    target_id: str
    requested_support_length: int
    actual_support_length: int
    clean_score: float | None
    poisoned_score: float | None
    poisoned_marked_incorrect_score: float | None
    poison_delta: float | None
    marked_incorrect_delta: float | None
    changed_by_poison: bool | None
    changed_by_marked_incorrect: bool | None
