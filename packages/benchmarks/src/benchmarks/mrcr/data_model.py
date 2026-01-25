from pydantic import BaseModel


class Score(BaseModel):
    index: int | str | None = None
    score: float
    context_length: int
