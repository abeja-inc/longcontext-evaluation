from abc import ABC, abstractmethod


class BaseReasoningParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> tuple[str, str]: ...
