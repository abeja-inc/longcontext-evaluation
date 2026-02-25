from difflib import SequenceMatcher
from logging import getLogger

import pytest
from benchmarks.config import SubtaskConfig
from benchmarks.mrcr.evaluate.metrics import OpenAIMRCRMetrics
from benchmarks.mrcr.predict.data import OpenAIMRCROutput
from benchmarks.mrcr.settings import OpenAIMRCRSettings


def _make_subtask_config() -> SubtaskConfig:
    return SubtaskConfig.model_validate(
        {
            "name": "mrcr-subtask",
            "language": "english",
            "dataset_filepath": "dummy.jsonl",
            "output_filepath": "dummy_output.jsonl",
            "metric": "prefix_match_similarity",
            "inference_mode": "chat",
            "settings": {},
        }
    )


@pytest.fixture
def mrcr_settings() -> OpenAIMRCRSettings:
    return OpenAIMRCRSettings.model_validate({"metric_kwargs": {"sentinel": "value"}})


def _make_output(
    *, output_text: str, answer: str, prefix: str = "PREFIX:"
) -> OpenAIMRCROutput:
    return OpenAIMRCROutput.model_validate(
        {
            "id": "1",
            "input": "question",
            "answer": answer,
            "output": output_text,
            "context_length": 128,
            "random_string_to_prepend": prefix,
            "n_needles": 1,
            "desired_msg_index": 0,
            "total_messages": 1,
        }
    )


def test_prefix_match_similarity_returns_zero_for_empty_output() -> None:
    metrics = OpenAIMRCRMetrics(logger=getLogger(__name__))
    output = _make_output(output_text="", answer="PREFIX:reference")

    score = metrics._prefix_match_similarity(
        output=output,
        default_error_message="__ERROR__",
    )

    assert score == 0


def test_prefix_match_similarity_returns_zero_for_default_error_message() -> None:
    metrics = OpenAIMRCRMetrics(logger=getLogger(__name__))
    output = _make_output(output_text="__ERROR__", answer="PREFIX:reference")

    score = metrics._prefix_match_similarity(
        output=output,
        default_error_message="__ERROR__",
    )

    assert score == 0


def test_prefix_match_similarity_returns_zero_when_output_does_not_start_with_prefix() -> (
    None
):
    metrics = OpenAIMRCRMetrics(logger=getLogger(__name__))
    output = _make_output(output_text="WRONG:prediction", answer="PREFIX:reference")

    score = metrics._prefix_match_similarity(
        output=output,
        default_error_message="__ERROR__",
    )

    assert score == 0


def test_prefix_match_similarity_returns_one_for_identical_suffix_after_prefix() -> (
    None
):
    metrics = OpenAIMRCRMetrics(logger=getLogger(__name__))
    output = _make_output(
        output_text="PREFIX:identical suffix",
        answer="PREFIX:identical suffix",
    )

    score = metrics._prefix_match_similarity(
        output=output,
        default_error_message="__ERROR__",
    )

    assert score == 1.0


def test_prefix_match_similarity_returns_sequence_matcher_ratio_for_different_suffix() -> (
    None
):
    metrics = OpenAIMRCRMetrics(logger=getLogger(__name__))
    output = _make_output(
        output_text="PREFIX:predicted suffix",
        answer="PREFIX:reference suffix",
    )

    expected = SequenceMatcher(None, "predicted suffix", "reference suffix").ratio()
    score = metrics._prefix_match_similarity(
        output=output,
        default_error_message="__ERROR__",
    )

    assert score == pytest.approx(expected)
    assert 0 < score < 1


def test_eval_prefix_match_similarity_forwards_metric_kwargs(
    monkeypatch: pytest.MonkeyPatch, mrcr_settings: OpenAIMRCRSettings
) -> None:
    metrics = OpenAIMRCRMetrics(logger=getLogger(__name__))
    output = _make_output(output_text="PREFIX:answer", answer="PREFIX:answer")
    captured_kwargs: dict[str, object] = {}

    def _fake_prefix_match_similarity(**kwargs: object) -> float:
        captured_kwargs.update(kwargs)
        return 0.123

    monkeypatch.setattr(
        metrics, "_prefix_match_similarity", _fake_prefix_match_similarity
    )

    score = metrics.eval_prefix_match_similarity(
        output=output,
        config=_make_subtask_config(),
        settings=mrcr_settings,
        default_error_message="__ERROR__",
    )

    assert score == 0.123
    assert captured_kwargs["sentinel"] == "value"
