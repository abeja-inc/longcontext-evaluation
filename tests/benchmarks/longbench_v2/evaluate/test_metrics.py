from logging import getLogger

from benchmarks.config import SubtaskConfig
from benchmarks.longbench_v2.evaluate.metrics import LongBenchV2Metrics
from benchmarks.longbench_v2.predict.data import LongBenchV2Output
from benchmarks.longbench_v2.settings import LongBenchV2Settings


DEFAULT_ERROR_MESSAGE = "INPUT_TOO_LONG"


def _make_output(*, output_text: str, answer: str) -> LongBenchV2Output:
    return LongBenchV2Output.model_validate(
        {
            "id": "1",
            "input": "question",
            "answer": answer,
            "output": output_text,
            "context_length": 128,
            "difficulty": "easy",
            "length": "short",
            "domain": "Single-Document QA",
            "sub_domain": "Academic",
        }
    )


def _make_config() -> SubtaskConfig:
    return SubtaskConfig.model_validate(
        {
            "name": "longbench-v2-subtask",
            "language": "english",
            "dataset_filepath": "dummy.jsonl",
            "output_filepath": "dummy_output.jsonl",
            "metric": "parsed_answer_match",
            "inference_mode": "chat",
            "settings": {},
        }
    )


def _make_settings(metric_kwargs: dict[str, object]) -> LongBenchV2Settings:
    return LongBenchV2Settings.model_construct(metric_kwargs=metric_kwargs)


def test_parsed_answer_match_valid_extraction_with_parenthesized_label() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="The correct answer is (A)", answer="A")

    score = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
    )

    assert score == 1.0


def test_parsed_answer_match_valid_extraction_with_plain_label() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="The correct answer is B", answer="B")

    score = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
    )

    assert score == 1.0


def test_parsed_answer_match_accepts_decorated_output_with_asterisks() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="*The correct answer is (C)*", answer="C")

    score = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
    )

    assert score == 1.0


def test_parsed_answer_match_empty_output_returns_policy_values() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="", answer="A")

    score_without_compensation = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
        compensate_missing=False,
    )
    score_with_compensation = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
        compensate_missing=True,
    )

    assert score_without_compensation == 0.0
    assert score_with_compensation == 0.25


def test_parsed_answer_match_default_error_message_returns_policy_values() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text=DEFAULT_ERROR_MESSAGE, answer="A")

    score_without_compensation = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
        compensate_missing=False,
    )
    score_with_compensation = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
        compensate_missing=True,
    )

    assert score_without_compensation == 0.0
    assert score_with_compensation == 0.25


def test_parsed_answer_match_parse_failure_returns_policy_values() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="Answer might be E", answer="A")

    score_without_compensation = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
        compensate_missing=False,
    )
    score_with_compensation = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
        compensate_missing=True,
    )

    assert score_without_compensation == 0.0
    assert score_with_compensation == 0.25


def test_parsed_answer_match_invalid_gold_answer_returns_zero() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="The correct answer is (A)", answer="Z")

    score = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
    )

    assert score == 0.0


def test_parsed_answer_match_label_mismatch_returns_zero() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="The correct answer is (A)", answer="B")

    score = metrics._parsed_answer_match(
        output=output,
        default_error_message=DEFAULT_ERROR_MESSAGE,
    )

    assert score == 0.0


def test_eval_parsed_answer_match_uses_settings_metric_kwargs() -> None:
    metrics = LongBenchV2Metrics(logger=getLogger(__name__))
    output = _make_output(output_text="", answer="A")
    settings = _make_settings(metric_kwargs={"compensate_missing": True})

    score = metrics.eval_parsed_answer_match(
        output=output,
        config=_make_config(),
        settings=settings,
        default_error_message=DEFAULT_ERROR_MESSAGE,
    )

    assert score == 0.25
