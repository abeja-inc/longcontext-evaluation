from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar


ScoreT = TypeVar("ScoreT")


class BaseRecordScorer(ABC, Generic[ScoreT]):
    """
    ベンチマークごとの採点処理の共通インターフェース。

    新しいベンチマークを追加する際は、このクラスを継承して
    `score_records()` を実装すればOK。
    """

    name: str

    @abstractmethod
    def score_records(self, records: list[dict[str, Any]]) -> ScoreT:
        raise NotImplementedError
