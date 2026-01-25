from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRecordScorer(ABC):
    name: str

    @abstractmethod
    def score_records(self, records: list[dict[str, Any]]) -> Any:
        raise NotImplementedError
