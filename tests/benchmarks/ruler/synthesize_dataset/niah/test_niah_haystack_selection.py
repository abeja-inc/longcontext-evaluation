import importlib
import logging
import random
import re
import sys
import types
from pathlib import Path
from typing import Any


NEEDLE_SENTENCE_PATTERN = re.compile(
    r"One of the special magic [^\s]+ for [^\n.]+ is: [^\n.]+\."
)


def _install_tokenizer_stubs() -> None:
    tiktoken_module = types.ModuleType("tiktoken")

    class _DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(c) for c in text]

    def _encoding_for_model(_: str) -> _DummyTokenizer:
        return _DummyTokenizer()

    tiktoken_module.encoding_for_model = _encoding_for_model
    tiktoken_module.get_encoding = _encoding_for_model
    sys.modules.setdefault("tiktoken", tiktoken_module)

    transformers_module = types.ModuleType("transformers")
    transformers_module.AutoTokenizer = type("AutoTokenizer", (), {})
    sys.modules.setdefault("transformers", transformers_module)


def _build_generator(*, tmp_path: Path, type_haystack: str) -> Any:
    _install_tokenizer_stubs()
    config_module = importlib.import_module(
        "benchmarks.ruler.synthesize_dataset.config"
    )
    niah_module = importlib.import_module("benchmarks.ruler.synthesize_dataset.niah")

    config = config_module.NIAHSynthesisConfig(
        save_dirpath=tmp_path,
        task="niah",
        subset="unit-test",
        paulgraham_essay_path=tmp_path / "unused_essay.json",
        type_haystack=type_haystack,
        type_needle_k="numbers",
        type_needle_v="numbers",
        num_needle_k=2,
        num_needle_v=2,
        num_needle_q=1,
        tokenizer_type="tiktoken",
    )
    generator = niah_module.NIAHDatasetGenerator(
        config=config, logger=logging.getLogger(__name__)
    )
    generator._initialize_haystack()
    return generator


def _extract_context_body(user_prompt: str) -> str:
    match = re.search(
        r"I will quiz you about the [^\n]+ afterwards\.\n(?P<context>.*)\nWhat are all the special magic",
        user_prompt,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("context")


def _extract_needle_candidates(text: str) -> list[str]:
    return NEEDLE_SENTENCE_PATTERN.findall(text)


def test_gen_one_sample_noise_haystack_keeps_noise_as_main_component(
    tmp_path: Path,
) -> None:
    generator = _build_generator(tmp_path=tmp_path, type_haystack="noise")
    random.seed(2024)

    content, extra_fields = generator._gen_one_sample(sample_index=0, num_units=30)

    target_needles = list(extra_fields["needles"])
    context_body = _extract_context_body(content.user_prompt)
    lines = context_body.splitlines()
    needle_candidates = _extract_needle_candidates(context_body)
    base_sentence = generator.haystack_source[0]

    assert (
        len(target_needles)
        == generator.config.num_needle_k * generator.config.num_needle_v
    )
    assert len(lines) == 30 + len(target_needles)
    assert lines.count(base_sentence) == 30
    assert needle_candidates == target_needles


def test_gen_one_sample_needle_haystack_keeps_targets_identifiable_as_inserted(
    tmp_path: Path,
) -> None:
    generator = _build_generator(tmp_path=tmp_path, type_haystack="needle")
    random.seed(2024)

    content, extra_fields = generator._gen_one_sample(sample_index=0, num_units=40)

    target_needles = list(extra_fields["needles"])
    context_body = _extract_context_body(content.user_prompt)
    needle_candidates = _extract_needle_candidates(context_body)

    assert target_needles
    assert set(target_needles).issubset(set(needle_candidates))
    assert len(needle_candidates) > len(target_needles)
    assert set(needle_candidates) != set(target_needles)
