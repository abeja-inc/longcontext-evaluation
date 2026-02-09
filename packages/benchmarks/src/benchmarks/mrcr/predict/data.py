from ..._core.predict.data import Input, Output


class OpenAIMRCRInput(Input):
    prompt: str
    answer: str
    random_string_to_prepend: str
    n_needles: int
    desired_msg_index: int
    total_messages: int


class OpenAIMRCROutput(Output):
    answer: str
    random_string_to_prepend: str
    n_needles: int
    desired_msg_index: int
    total_messages: int
