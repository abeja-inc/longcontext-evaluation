import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "packages/benchmarks/src"))

from benchmarks.mrcr.scoring import Grader, MrcrScorer  # noqa: E402


class TestGrader(unittest.TestCase):
    def test_compute_rejects_max_context_message(self):
        grader = Grader()
        score = grader.compute("Max context tokens exceeded", "ref", "prefix")
        self.assertEqual(score, 0.0)

    def test_compute_requires_prefix(self):
        grader = Grader()
        score = grader.compute("answer", "prefixanswer", "prefix")
        self.assertEqual(score, 0.0)

    def test_compute_scores_after_prefix(self):
        grader = Grader()
        score = grader.compute("prefixanswer", "prefixanswer", "prefix")
        self.assertEqual(score, 1.0)


class TestMrcrScorer(unittest.TestCase):
    def test_score_builds_indexed_scores(self):
        scorer = MrcrScorer()
        scores = scorer.score_records(
            [
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
        )
        self.assertEqual([s.index for s in scores], [0, 1])
        self.assertEqual([s.context_length for s in scores], [128, 256])
        self.assertTrue(all(s.score == 1.0 for s in scores))


if __name__ == "__main__":
    unittest.main()
