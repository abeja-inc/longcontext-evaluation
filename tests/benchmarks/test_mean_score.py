from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
import sys

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_SRC = ROOT / "packages" / "benchmarks" / "src"
MEAN_SCORE_PATH = (
    BENCHMARKS_SRC
    / "benchmarks"
    / "_core"
    / "evaluate"
    / "mean_score.py"
)

sys.path.insert(0, str(BENCHMARKS_SRC))

benchmarks_pkg = ModuleType("benchmarks")
benchmarks_pkg.__path__ = [str(BENCHMARKS_SRC / "benchmarks")]
sys.modules.setdefault("benchmarks", benchmarks_pkg)

core_pkg = ModuleType("benchmarks._core")
core_pkg.__path__ = [str(BENCHMARKS_SRC / "benchmarks" / "_core")]
sys.modules.setdefault("benchmarks._core", core_pkg)

evaluate_pkg = ModuleType("benchmarks._core.evaluate")
evaluate_pkg.__path__ = [
    str(BENCHMARKS_SRC / "benchmarks" / "_core" / "evaluate")
]
sys.modules.setdefault("benchmarks._core.evaluate", evaluate_pkg)

spec = spec_from_file_location(
    "benchmarks._core.evaluate.mean_score",
    MEAN_SCORE_PATH,
)
mean_score = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mean_score)

Bin = mean_score.Bin
mean_score_by_group_and_context_bin = mean_score.mean_score_by_group_and_context_bin


def test_mean_score_by_group_and_context_bin_empty_rows():
    result = mean_score_by_group_and_context_bin(
        [],
        group_by="task",
        context_bins=[Bin(upper=512, label="short")],
    )

    assert result == []
