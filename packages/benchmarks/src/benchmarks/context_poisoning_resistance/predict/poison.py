import random
import re


_SYMBOL_GROUPS: dict[str, list[str]] = {
    "+": ["+", "＋"],
    "-": ["-", "ー"],
    "÷": ["/", "÷"],
    "×": ["x", "×", "*"],
}
_FLIP_GROUP = {"+": "-", "-": "+", "÷": "×", "×": "÷"}
_CHAR_TO_GROUP = {
    char: group for group, chars in _SYMBOL_GROUPS.items() for char in chars
}
_PATTERN = re.compile("|".join(map(re.escape, _CHAR_TO_GROUP.keys())))


def poison_answer_text(text: str, seed: int) -> str:
    rng = random.Random(seed)

    def repl(match: re.Match[str]) -> str:
        group = _CHAR_TO_GROUP[match.group(0)]
        flipped = _FLIP_GROUP[group]
        return rng.choice(_SYMBOL_GROUPS[flipped])

    return _PATTERN.sub(repl, text)
