from .base import BaseReasoningParser


class OpenAIGPTOSSReasoningParser(BaseReasoningParser):
    def parse(self, text: str) -> tuple[str, str]:
        reasoning_content, sep, content = text.rpartition("assistantfinal")
        return (reasoning_content + sep).strip(), content.strip()
