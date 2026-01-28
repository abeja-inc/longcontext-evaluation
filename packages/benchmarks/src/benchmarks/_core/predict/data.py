from dataclasses import dataclass, field


@dataclass(frozen=True)
class Output:
    input: str
    answer: object
    output: str
    context_length: int

    index: int | str | None = None
    output_reasoning: str | None = None
    extra: dict[str, object] = field(default_factory=dict)
