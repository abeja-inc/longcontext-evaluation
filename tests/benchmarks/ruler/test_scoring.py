import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "packages/benchmarks/src"))

from benchmarks.ruler.scoring import AllStringMatcher, PartStringMatcher, RulerScorer  # noqa: E402


class TestStringMatchers(unittest.TestCase):
    def test_part_string_matcher_scores_any_match(self):
        matcher = PartStringMatcher()
        score = matcher.compute(preds=["answer foo"], refs=[["foo", "bar"]])
        self.assertEqual(score, 100.0)

    def test_all_string_matcher_requires_all_matches(self):
        matcher = AllStringMatcher()
        score = matcher.compute(preds=["answer foo"], refs=[["foo", "bar"]])
        self.assertEqual(score, 50.0)


class TestRulerScorer(unittest.TestCase):
    def test_score_returns_per_sample_scores(self):
        scorer = RulerScorer(metric="part")
        scores = scorer.score_records(
            [
                {"prediction": "foo", "answer": ["foo"], "target_context_length": 4096},
                {"prediction": "bar", "answer": ["bar"], "target_context_length": 8192},
            ]
        )
        self.assertEqual([s.index for s in scores], [0, 1])
        self.assertEqual([s.context_length for s in scores], [4096, 8192])
        self.assertTrue(all(s.score == 100.0 for s in scores))


if __name__ == "__main__":
    unittest.main()
