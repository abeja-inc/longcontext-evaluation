from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LLM_INFERENCE_SRC = ROOT / "packages" / "llm_inference" / "src"
BENCHMARKS_SRC = ROOT / "packages" / "benchmarks" / "src"

for path in (LLM_INFERENCE_SRC, BENCHMARKS_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if "tqdm" not in sys.modules:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable=None, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

if "rapidfuzz" not in sys.modules:
    rapidfuzz_module = types.ModuleType("rapidfuzz")
    distance_module = types.ModuleType("rapidfuzz.distance")

    class DummyLCSseq:
        @staticmethod
        def similarity(a: str, b: str) -> int:
            return len(a) if a == b else 0

    distance_module.LCSseq = DummyLCSseq
    rapidfuzz_module.distance = distance_module
    sys.modules["rapidfuzz"] = rapidfuzz_module
    sys.modules["rapidfuzz.distance"] = distance_module
