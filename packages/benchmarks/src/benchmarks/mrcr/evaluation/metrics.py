from difflib import SequenceMatcher


class Grader:
    def compute(self, pred: str, ref: str, random_string_to_prepend: str) -> float:
        if pred == "Max context tokens exceeded":
            return 0
        if not pred.startswith(random_string_to_prepend):
            return 0

        pred = pred.removeprefix(random_string_to_prepend)
        ref = ref.removeprefix(random_string_to_prepend)

        return float(SequenceMatcher(None, pred, ref).ratio())

    def grade(
        self, preds: list[str], refs: list[str], random_string_to_prepends: list[str]
    ) -> float:
        grades = [
            self.compute(pred, ref, random_string_to_prepend)
            for pred, ref, random_string_to_prepend in zip(
                preds, refs, random_string_to_prepends, strict=True
            )
        ]

        return sum(grades) / len(grades)
