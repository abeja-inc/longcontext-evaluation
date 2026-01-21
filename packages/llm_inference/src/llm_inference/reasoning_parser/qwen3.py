from .base import BaseReasoningParser


class Qwen3ReasoningParser(BaseReasoningParser):
    def parse(self, text: str) -> tuple[str, str]:
        reasoning_content, sep, content = text.rpartition("</think>")
        return (reasoning_content + sep).strip(), content.strip()
