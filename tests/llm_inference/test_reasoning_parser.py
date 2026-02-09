from __future__ import annotations

import pytest

from llm_inference.reasoning_parser import resolve_reasoning_parser
from llm_inference.reasoning_parser.openai_gptoss import OpenAIGPTOSSReasoningParser
from llm_inference.reasoning_parser.qwen3 import Qwen3ReasoningParser


def test_resolve_reasoning_parser_variants() -> None:
    assert resolve_reasoning_parser(None) is None
    assert isinstance(resolve_reasoning_parser("qwen3"), Qwen3ReasoningParser)
    parser = OpenAIGPTOSSReasoningParser()
    assert resolve_reasoning_parser(parser) is parser
    with pytest.raises(ValueError, match="Unknown reasoning parser"):
        resolve_reasoning_parser("unknown")


def test_qwen3_parser_boundaries() -> None:
    parser = Qwen3ReasoningParser()
    reasoning, content = parser.parse("thinking</think>answer")
    assert reasoning == "thinking</think>"
    assert content == "answer"

    reasoning, content = parser.parse("no delimiter here")
    assert reasoning == ""
    assert content == "no delimiter here"


def test_openai_gptoss_parser_boundaries() -> None:
    parser = OpenAIGPTOSSReasoningParser()
    reasoning, content = parser.parse("analysis assistantfinal answer")
    assert reasoning == "analysis assistantfinal"
    assert content == "answer"

    reasoning, content = parser.parse("plain output")
    assert reasoning == ""
    assert content == "plain output"
