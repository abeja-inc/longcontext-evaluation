import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "packages/benchmarks/src"))

from benchmarks.longbench_v2.scoring import evaluate_records, extract_answer  # noqa: E402


class TestExtractAnswer(unittest.TestCase):
    def test_extract_answer_with_parentheses(self):
        self.assertEqual(
            extract_answer("The correct answer is (B)."), "B"
        )

    def test_extract_answer_without_parentheses(self):
        self.assertEqual(
            extract_answer("The correct answer is C"), "C"
        )

    def test_extract_answer_missing(self):
        self.assertIsNone(extract_answer("No answer here."))


class TestEvaluateRecords(unittest.TestCase):
    def test_evaluate_records_counts_accuracy(self):
        records = [
            {
                "answer": "A",
                "prediction": "The correct answer is (A)",
                "difficulty": "easy",
                "length": "short",
                "id": "s1",
            },
            {
                "answer": "B",
                "prediction": "The correct answer is (C)",
                "difficulty": "hard",
                "length": "long",
                "id": "s2",
            },
        ]
        metrics, rows = evaluate_records(records)
        self.assertEqual(metrics["overall_n"], 2)
        self.assertEqual(metrics["overall_acc"], 50.0)
        self.assertEqual(metrics["easy_acc"], 100.0)
        self.assertEqual(metrics["hard_acc"], 0.0)
        self.assertEqual(metrics["short_n"], 1)
        self.assertEqual(metrics["long_n"], 1)
        self.assertEqual(rows[0]["norm_prediction"], "A")


if __name__ == "__main__":
    unittest.main()
