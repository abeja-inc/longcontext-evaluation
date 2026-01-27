import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "packages/benchmarks/src"))

from benchmarks.mrcr.evaluation.evaluator import (  # noqa: E402
    MRCREvaluator,
    TaskSetting,
)


class TestMRCREvaluator(unittest.TestCase):
    def test_run_builds_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir) / "preds"
            task_dir = prediction_dir / "task1"
            task_dir.mkdir(parents=True, exist_ok=True)
            pred_file = task_dir / "subset.jsonl"
            records = [
                {
                    "prediction": "prefixfoo",
                    "answer": ["prefixfoo"],
                    "random_string_to_prepend": "prefix",
                    "target_context_length": 128,
                },
                {
                    "prediction": "prefixbar",
                    "answer": ["prefixbar"],
                    "random_string_to_prepend": "prefix",
                    "target_context_length": 256,
                },
            ]
            pred_file.write_text(
                "\n".join(json.dumps(r) for r in records), encoding="utf-8"
            )

            evaluator = MRCREvaluator(
                tasks=[
                    TaskSetting(
                        task="task1",
                        metric="default",
                        filenames=["subset.jsonl"],
                    )
                ]
            )
            output_path = prediction_dir / "summary.json"

            summary = evaluator.run(
                prediction_dir=prediction_dir, output_path=output_path
            )

            self.assertTrue(output_path.exists())
            self.assertIsNotNone(summary)
            self.assertEqual(summary["results"][0]["task"], "task1")
            subsets = summary["results"][0]["subsets"]
            self.assertEqual(subsets[0]["subset_name"], "subset")
            self.assertEqual(len(subsets[0]["score"]), 2)


if __name__ == "__main__":
    unittest.main()
