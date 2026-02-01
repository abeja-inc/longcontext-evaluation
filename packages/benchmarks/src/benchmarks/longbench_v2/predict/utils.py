from string import Template

from .data import LongBenchV2Input


def build_input_prompt(
    input: LongBenchV2Input,
    prompt_templates: dict[str, Template],
    rag_topn: int = 0,
    cot: bool = False,
    no_context: bool = False,
) -> tuple[str, Template | None]:
    # Document
    if rag_topn > 0 and input.retrieved_context:
        template = prompt_templates["zero_shot_rag"]
        chunks = sorted(
            [chunk.model_dump() for chunk in input.retrieved_context],
            key=lambda x: x.get("c_idx", 0),
        )[:rag_topn]
        document = "\n\n".join(
            [
                f"Retrieved chunk {i + 1}: {c.get('content', '')}"
                for i, c in enumerate(chunks)
            ]
        )
    else:
        document = input.context
        if no_context:
            template = prompt_templates["zero_shot_no_context"]
        elif cot:
            template = prompt_templates["zero_shot_cot"]
        else:
            template = prompt_templates["zero_shot"]

    # Question & choices
    question = input.question
    choice_A = input.choice_A
    choice_B = input.choice_B
    choice_C = input.choice_C
    choice_D = input.choice_D

    # Build prompt
    prompt_text = template.safe_substitute(
        {
            "DOC": document.strip(),
            "Q": question.strip(),
            "C_A": choice_A.strip(),
            "C_B": choice_B.strip(),
            "C_C": choice_C.strip(),
            "C_D": choice_D.strip(),
        }
    )

    # make cot template
    if cot:
        cot_template = Template(
            prompt_templates["zero_shot_cot_ans"].safe_substitute(
                {
                    "Q": question.strip(),
                    "C_A": choice_A.strip(),
                    "C_B": choice_B.strip(),
                    "C_C": choice_C.strip(),
                    "C_D": choice_D.strip(),
                }
            )
        )
    else:
        cot_template = None

    return prompt_text, cot_template
