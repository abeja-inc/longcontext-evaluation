import importlib.util
import sys
import types
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BENCHMARKS_SRC = ROOT_DIR / "packages" / "benchmarks" / "src" / "benchmarks"
SYNTH_DATASET_DIR = BENCHMARKS_SRC / "ruler" / "synthesize_dataset"

sys.path.append(str(ROOT_DIR / "packages" / "llm_inference" / "src"))


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmarks_pkg = types.ModuleType("benchmarks")
benchmarks_pkg.__path__ = [str(BENCHMARKS_SRC)]
sys.modules.setdefault("benchmarks", benchmarks_pkg)

ruler_pkg = types.ModuleType("benchmarks.ruler")
ruler_pkg.__path__ = [str(BENCHMARKS_SRC / "ruler")]
sys.modules.setdefault("benchmarks.ruler", ruler_pkg)

synth_pkg = types.ModuleType("benchmarks.ruler.synthesize_dataset")
synth_pkg.__path__ = [str(SYNTH_DATASET_DIR)]
sys.modules.setdefault("benchmarks.ruler.synthesize_dataset", synth_pkg)


token_counter_module = types.ModuleType("llm_inference.token_counter")


class ImportTokenCounter:
    def __init__(self, **_kwargs: object) -> None:
        return


token_counter_module.TokenCounter = ImportTokenCounter
sys.modules["llm_inference.token_counter"] = token_counter_module

numpy_module = types.ModuleType("numpy")


class _RandomStub:
    def seed(self, *_args: object, **_kwargs: object) -> None:
        return


numpy_module.random = _RandomStub()
sys.modules["numpy"] = numpy_module

config_module = _load_module(
    "benchmarks.ruler.synthesize_dataset.config", SYNTH_DATASET_DIR / "config.py"
)
data_model_module = _load_module(
    "benchmarks.ruler.synthesize_dataset.data_model",
    SYNTH_DATASET_DIR / "data_model.py",
)
base_module = _load_module(
    "benchmarks.ruler.synthesize_dataset.base", SYNTH_DATASET_DIR / "base.py"
)

BaseDatasetGenerator = base_module.BaseDatasetGenerator
BaseSynthesisConfig = config_module.BaseSynthesisConfig
BaseDatasetSchema = data_model_module.BaseDatasetSchema
Content = data_model_module.Content


class StubTokenCounter:
    def count_tokens(self, value, *_args, **_kwargs) -> int:
        if hasattr(value, "messages"):
            return sum(len(message.content) for message in value.messages)
        return len(value)


class DummyDatasetGenerator(BaseDatasetGenerator[BaseDatasetSchema, BaseSynthesisConfig]):
    @property
    def SCHEMA_CLASS(self) -> type[BaseDatasetSchema]:
        return BaseDatasetSchema

    def __init__(self, config: BaseSynthesisConfig) -> None:
        super().__init__(config=config, logger=None)
        self.token_counter = StubTokenCounter()

    def _gen_one_sample(
        self, sample_index: int, num_units: int, **_kwargs: object
    ) -> tuple[Content, dict[str, object]]:
        content = Content(
            user_prompt="u" * (num_units * 10),
            answer_prefix="",
            outputs=[""],
        )
        return content, {"target_depth_percent": 0.0}


def _make_config(tmp_path: Path, max_new_tokens: int) -> BaseSynthesisConfig:
    return BaseSynthesisConfig(
        save_dirpath=tmp_path,
        task="task",
        subset="subset",
        task_prompt_template="prompt {context}",
        answer_prefix_template="answer",
        max_new_tokens=max_new_tokens,
        initial_units=4,
    )


def test_optimal_units_picks_largest_before_limit(tmp_path: Path) -> None:
    config = _make_config(tmp_path, max_new_tokens=5)
    generator = DummyDatasetGenerator(config)

    max_units = generator._optimal_units(max_context_length=95)

    assert max_units == 9


def test_optimal_units_returns_one_when_context_too_small(tmp_path: Path) -> None:
    config = _make_config(tmp_path, max_new_tokens=10)
    generator = DummyDatasetGenerator(config)

    max_units = generator._optimal_units(max_context_length=10)

    assert max_units == 1
