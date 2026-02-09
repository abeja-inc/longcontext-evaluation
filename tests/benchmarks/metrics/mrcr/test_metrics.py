import logging
from pathlib import Path

from benchmarks.config import SubtaskConfig
from benchmarks.mrcr.evaluate.metrics import OpenAIMRCRMetrics
from benchmarks.mrcr.predict.data import OpenAIMRCROutput
from benchmarks.mrcr.settings import OpenAIMRCRSettings


def test_prefix_match_similarity_returns_zero_when_prefix_missing() -> None:
    metrics = OpenAIMRCRMetrics(logger=logging.getLogger(__name__))
    output = OpenAIMRCROutput(
        id="sample",
        input="prompt",
        answer="PREFIX_expected answer",
        output="unexpected answer",
        context_length=10,
        random_string_to_prepend="PREFIX_",
        n_needles=1,
        desired_msg_index=0,
        total_messages=1,
    )
    config = SubtaskConfig(
        name="mrcr",
        language="english",
        dataset_filepath=Path("dataset.jsonl"),
        output_filepath=Path("output.jsonl"),
        metric="prefix_match_similarity",
        inference_mode="completion",
        settings=None,
    )
    settings = OpenAIMRCRSettings()

    score = metrics.eval_prefix_match_similarity(
        output=output,
        config=config,
        settings=settings,
        default_error_message="ERROR",
    )

    assert score == 0.0
