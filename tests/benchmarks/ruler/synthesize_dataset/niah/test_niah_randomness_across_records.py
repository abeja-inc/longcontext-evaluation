import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest


if "nltk" not in sys.modules:
    import types

    fake_nltk = types.SimpleNamespace(download=lambda *_args, **_kwargs: None)
    fake_tokenize = types.SimpleNamespace(
        sent_tokenize=lambda text: [
            s.strip() + "." for s in text.split(".") if s.strip()
        ]
    )
    sys.modules["nltk"] = fake_nltk
    sys.modules["nltk.tokenize"] = fake_tokenize

if "tiktoken" not in sys.modules:
    import types

    sys.modules["tiktoken"] = types.SimpleNamespace(
        encoding_for_model=lambda *_args, **_kwargs: None,
        get_encoding=lambda *_args, **_kwargs: None,
    )

if "transformers" not in sys.modules:
    import types

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> Any:
            return None

    sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=_AutoTokenizer)

if "numpy" not in sys.modules:
    import types

    sys.modules["numpy"] = types.SimpleNamespace(
        random=types.SimpleNamespace(seed=lambda *_args, **_kwargs: None)
    )

if "wonderwords" not in sys.modules:
    import types

    fake_random_word = types.SimpleNamespace(
        _get_words_from_text_file=lambda _name: ["alpha", "beta", "gamma"]
    )
    sys.modules["wonderwords"] = types.SimpleNamespace(random_word=fake_random_word)

ROOT_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT_DIR / "packages/benchmarks/src/benchmarks/ruler"))
sys.path.insert(0, str(ROOT_DIR / "packages/llm_inference/src"))

from synthesize_dataset.config import NIAHSynthesisConfig
from synthesize_dataset.niah.niah import NIAHDatasetGenerator


class DummyTokenCounter:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass


def _build_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, random_seed: int = 2024
) -> NIAHDatasetGenerator:
    monkeypatch.setattr("synthesize_dataset.base.TokenCounter", DummyTokenCounter)
    monkeypatch.setattr(
        "synthesize_dataset.niah.niah.sent_tokenize",
        lambda text: [s.strip() + "." for s in text.split(".") if s.strip()],
    )

    essay_path = tmp_path / "essay.json"
    essay_path.write_text(
        json.dumps(
            {
                "text": (
                    "Alpha writes code carefully. "
                    "Beta reviews each change thoroughly. "
                    "Gamma validates assumptions before shipping. "
                    "Delta keeps tests deterministic and clear. "
                    "Epsilon documents motivations for future readers. "
                )
            }
        ),
        encoding="utf-8",
    )

    config = NIAHSynthesisConfig(
        save_dirpath=tmp_path,
        task="ruler",
        subset="niah_randomness",
        paulgraham_essay_path=essay_path,
        tokenizer_name_or_path="gpt-4o",
        tokenizer_type="tiktoken",
        random_seed=random_seed,
        type_haystack="essay",
        type_needle_k="words",
        type_needle_v="words",
        num_needle_k=3,
        num_needle_v=2,
        num_needle_q=1,
        task_prompt_template="CTX_START\n{context}\nCTX_END\nQUESTION:{query}",
        answer_prefix_template="ANS:{query}",
    )

    generator = NIAHDatasetGenerator(config=config, logger=logging.getLogger(__name__))
    generator.haystack_source = []
    generator._initialize_haystack()
    generator.words = [
        "amber-fox",
        "blue-owl",
        "crimson-hawk",
        "daring-whale",
        "emerald-wolf",
        "frozen-eagle",
        "golden-lark",
        "hazel-shark",
    ]

    return generator


def _extract_context(user_prompt: str) -> str:
    return user_prompt.split("CTX_START\n", 1)[1].split("\nCTX_END", 1)[0]


def test_niah_randomness_is_reflected_across_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _build_generator(tmp_path=tmp_path, monkeypatch=monkeypatch)

    num_samples = 40
    records: list[dict[str, Any]] = []
    for sample_index in range(num_samples):
        content, extra_fields = generator._gen_one_sample(
            sample_index=sample_index,
            num_units=120,
        )
        records.append({"content": content, "extra": extra_fields})

    needle_query_pairs = {
        (tuple(rec["extra"]["needles"]), rec["extra"]["query"]) for rec in records
    }
    assert len(needle_query_pairs) >= 3

    depth_patterns = {tuple(rec["extra"]["target_depth_percent"]) for rec in records}
    assert len(depth_patterns) >= 3

    all_depth_values = {
        depth for rec in records for depth in rec["extra"]["target_depth_percent"]
    }
    assert len(all_depth_values) > 1

    contexts = [_extract_context(rec["content"].user_prompt) for rec in records]
    head_snippets = {ctx[:120] for ctx in contexts}
    mid_snippets = {
        ctx[max((len(ctx) // 2) - 60, 0) : max((len(ctx) // 2) - 60, 0) + 120]
        for ctx in contexts
    }
    assert len(head_snippets) >= 3
    assert len(mid_snippets) >= 3
