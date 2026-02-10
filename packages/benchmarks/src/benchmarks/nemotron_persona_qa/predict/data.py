from pydantic import AliasChoices, Field

from ..._core.predict.data import Input, Output


class NemotronPersonaQAInput(Input):
    prompt: str
    answer: str
    num_personas: int = Field(validation_alias=AliasChoices("length"))


class NemotronPersonaQAOutput(Output):
    answer: str
    num_personas: int
