import logging
import sys
import types


sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

from benchmarks.context_poisoning_resistance.evaluate.table import (
    ContextPoisoningResistanceFilteredComparisonRow,
    ContextPoisoningResistanceFilteredLeaderBoardTableRow,
    ContextPoisoningResistanceOutputsTableRow,
)
from benchmarks.context_poisoning_resistance.runner import (
    ContextPoisoningResistanceRunner,
)


def _row(
    *,
    target_id: str,
    condition: str,
    score: float,
    stage1_target_is_correct: bool = True,
    support_length: int = 2,
    requested_support_length: int = 2,
) -> ContextPoisoningResistanceOutputsTableRow:
    return ContextPoisoningResistanceOutputsTableRow(
        model_name="model",
        id=f"{target_id}-{condition}",
        task="make_ten_jpn",
        subtask=f"{condition}_k{requested_support_length}",
        language="japanese",
        context_length=128,
        input="input",
        answer="10",
        score=score,
        output="output",
        output_reasoning=None,
        target_id=target_id,
        requested_support_ids=["s1", "s2"],
        support_ids=["s1", "s2"][:support_length],
        skipped_support_ids=["s1", "s2"][support_length:],
        condition=condition,
        requested_support_length=requested_support_length,
        support_length=support_length,
        used_stage1_support_count=support_length,
        target_numbers=["1", "2", "3", "4"],
        stage1_target_output="1+2+3+4=10",
        stage1_target_is_correct=stage1_target_is_correct,
        poisoned_support_outputs=[],
        final_is_correct=score >= 1.0,
    )


def test_filtered_tables_only_include_clean_solved_full_support_pairs() -> None:
    runner = ContextPoisoningResistanceRunner(logger=logging.getLogger(__name__))
    outputs = [
        _row(target_id="eligible", condition="clean", score=1.0),
        _row(target_id="eligible", condition="poisoned", score=0.0),
        _row(target_id="eligible", condition="poisoned_marked_incorrect", score=1.0),
        _row(
            target_id="stage1-fail",
            condition="clean",
            score=1.0,
            stage1_target_is_correct=False,
        ),
        _row(target_id="stage1-fail", condition="poisoned", score=0.0),
        _row(
            target_id="stage1-fail",
            condition="poisoned_marked_incorrect",
            score=1.0,
        ),
        _row(target_id="clean-fail", condition="clean", score=0.0),
        _row(target_id="clean-fail", condition="poisoned", score=0.0),
        _row(target_id="clean-fail", condition="poisoned_marked_incorrect", score=1.0),
        _row(
            target_id="support-missing",
            condition="clean",
            score=1.0,
            support_length=1,
        ),
        _row(target_id="support-missing", condition="poisoned", score=0.0),
        _row(
            target_id="support-missing",
            condition="poisoned_marked_incorrect",
            score=1.0,
        ),
    ]

    tables = {
        table.name: table.rows for table in runner._make_additional_tables(outputs)
    }

    filtered_rows = tables[
        "context_poisoning_resistance_filtered_condition_comparison_table"
    ]
    assert filtered_rows == [
        ContextPoisoningResistanceFilteredComparisonRow(
            model_name="model",
            target_id="eligible",
            requested_support_length=2,
            clean_score=1.0,
            poisoned_score=0.0,
            poisoned_marked_incorrect_score=1.0,
            poison_drop=1.0,
            marked_incorrect_recovery=1.0,
            changed_by_poison=True,
            changed_by_marked_incorrect=True,
        )
    ]

    leaderboard_rows = tables["context_poisoning_resistance_filtered_leaderboard_table"]
    assert leaderboard_rows == [
        ContextPoisoningResistanceFilteredLeaderBoardTableRow(
            model_name="model",
            eligible_pairs=1,
            poisoned_accuracy=0.0,
            poisoned_marked_incorrect_accuracy=1.0,
            poison_drop=1.0,
            marked_incorrect_recovery=1.0,
        )
    ]
