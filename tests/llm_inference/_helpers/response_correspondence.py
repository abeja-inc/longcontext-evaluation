from llm_inference.data import Response


def assert_response_outputs_match_expected_order(
    responses: list[Response],
    expected_outputs: list[str],
) -> None:
    assert [response.outputs[0].content for response in responses] == expected_outputs


def expected_outputs_with_middle_too_long(
    *,
    inputs: list[str],
    default_too_long_input_error_message: str,
) -> list[str]:
    assert len(inputs) >= 3
    return [
        f"response:{inputs[0]}",
        default_too_long_input_error_message,
        f"response:{inputs[2]}",
    ]
