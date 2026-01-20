from .base import BaseReasoningParser
from .openai_gptoss import OpenAIGPTOSSReasoningParser
from .qwen3 import Qwen3ReasoningParser


_REGISTRY: dict[str, type[BaseReasoningParser]] = {
    "qwen3": Qwen3ReasoningParser,
    "openai_gptoss": OpenAIGPTOSSReasoningParser,
}

ReasoningParserLike = str | BaseReasoningParser | None


def resolve_reasoning_parser(parser: ReasoningParserLike) -> BaseReasoningParser | None:
    if parser is None:
        return None

    if isinstance(parser, BaseReasoningParser):
        return parser

    # str case
    try:
        return _REGISTRY[parser]()
    except KeyError as e:
        raise ValueError(
            f"Unknown reasoning parser: {parser}. Available: {list(_REGISTRY.keys())}"
        ) from e
