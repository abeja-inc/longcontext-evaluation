import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = ["domain", "sub_domain", "difficulty", "length"]


def analyze_fields(jsonl_path: str | Path) -> None:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(path)

    counters: dict[str, Counter[str]] = {f: Counter() for f in FIELDS}
    missing: dict[str, int] = {f: 0 for f in FIELDS}

    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {line_no}: {e}") from e

            total += 1

            for field in FIELDS:
                v = obj.get(field)
                if v is None:
                    missing[field] += 1
                else:
                    counters[field][str(v)] += 1

    print(f"Total records: {total}\n")

    for field in FIELDS:
        print(f"## `{field}`\n")
        print("| value | count |")
        print("|-------|-------|")

        for value, count in counters[field].most_common():
            print(f"| {value} | {count} |")

        if missing[field] > 0:
            print(f"\n> missing `{field}`: {missing[field]}\n")
        else:
            print("\n")


if __name__ == "__main__":
    analyze_fields(
        Path(
            "../datasets/benchmarks/longbench_v2/english/longbench_v2_with_token_count.jsonl"
        )
    )
