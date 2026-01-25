from pydantic import BaseModel


class Score(BaseModel):
    score: float
    context_length: int
