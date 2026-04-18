from string import Template

from llm_inference.data import Conversation, Message

from ..settings import ContextPoisoningResistanceSettings


def _render_question(numbers: list[str], question_template: Template) -> str:
    return question_template.safe_substitute(
        {
            "num_1": numbers[0],
            "num_2": numbers[1],
            "num_3": numbers[2],
            "num_4": numbers[3],
        }
    )


def _render_prefix(label: str, question_prefix_template: Template) -> str:
    label_map = {"correct": "正解", "incorrect": "不正解"}
    return question_prefix_template.safe_substitute({"bool": label_map[label]})


def build_stage1_conversation(
    *,
    numbers: list[str],
    settings: ContextPoisoningResistanceSettings,
) -> Conversation:
    templates = settings.prompt.load_templates()
    return Conversation.model_validate(
        {
            "messages": [
                {"role": "system", "content": templates["system_prompt"].template},
                {
                    "role": "user",
                    "content": _render_question(numbers, templates["question"]),
                },
            ]
        }
    )


def build_stage2_conversation(
    *,
    target_numbers: list[str],
    support_examples: list[tuple[list[str], str, str]],
    target_judge_label: str,
    settings: ContextPoisoningResistanceSettings,
) -> Conversation:
    templates = settings.prompt.load_templates()
    messages: list[Message] = [
        Message(role="system", content=templates["system_prompt"].template)
    ]

    for idx, (numbers, answer, judge_label) in enumerate(support_examples):
        prefix = ""
        if idx > 0:
            prefix = _render_prefix(judge_label, templates["question_prefix"])
        messages.append(
            Message(
                role="user",
                content=prefix + _render_question(numbers, templates["question"]),
            )
        )
        messages.append(Message(role="assistant", content=answer))

    messages.append(
        Message(
            role="user",
            content=_render_prefix(target_judge_label, templates["question_prefix"])
            + _render_question(target_numbers, templates["question"]),
        )
    )
    return Conversation(messages=messages)
