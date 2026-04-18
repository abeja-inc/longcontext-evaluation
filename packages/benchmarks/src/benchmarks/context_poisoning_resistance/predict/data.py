from typing import Literal

from ..._core.predict.data import Input, Output


Condition = Literal["clean", "poisoned", "poisoned_marked_incorrect"]


class ContextPoisoningResistanceInput(Input):
    target_id: str
    support_ids: list[str]
    condition: Condition
    language: Literal["japanese"]
    target_numbers: list[str]
    support_number_orders: dict[str, list[str]]
    judge_label: Literal["correct", "incorrect"]


class ContextPoisoningResistanceOutput(Output):
    answer: str
    target_id: str
    requested_support_ids: list[str]
    support_ids: list[str]
    skipped_support_ids: list[str]
    condition: Condition
    requested_support_length: int
    support_length: int
    used_stage1_support_count: int
    target_numbers: list[str]
    stage1_target_output: str
    stage1_target_is_correct: bool
    poisoned_support_outputs: list[str]
