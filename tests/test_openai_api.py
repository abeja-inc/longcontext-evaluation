import logging
import os

from dotenv import load_dotenv
from llm_inference.data import Conversation, Prompt
from llm_inference.openai_api import OpenAIGenerator


# -----------------------------
# 1) logger
# -----------------------------
def build_logger() -> logging.Logger:
    logger = logging.getLogger("smoke_test_openai_generator")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def main() -> None:
    logger = build_logger()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    model = "gpt-4o-2024-11-20"

    gen = OpenAIGenerator(
        model=model,
        max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "8192")),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "128")),
        logger=logger,
        api_key=api_key,
    )

    # -------------------------
    # A) _count_tokens
    # -------------------------
    logger.info("=== (A) _count_tokens ===")
    short_prompt = Prompt.model_validate({"prompt": "Hello! How are you?"})
    tok = gen._count_tokens(input=short_prompt)
    logger.info("tokens(short_text) = %s", tok)

    short_conversation = Conversation.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hello! How are you?"},
                {"role": "assistant", "content": "I'm fine, thank you. How about you?"},
                {"role": "user", "content": "I'm good, thanks."},
            ]
        }
    )
    tok2 = gen._count_tokens(input=short_conversation)
    logger.info("tokens(conv) = %s", tok2)

    # -------------------------
    # B) _is_over_context_length
    # -------------------------
    logger.info("=== (B) _is_over_context_length ===")

    over = gen._is_over_context_length(
        input=short_prompt,
        max_context_length=50,  # わざと小さく
        max_output_tokens=gen.max_output_tokens,
        buffer_tokens=100,
    )
    logger.info("over_context prompt (expect True) = %s", over)

    ok = gen._is_over_context_length(
        input=short_prompt,
        max_context_length=5000,
        max_output_tokens=gen.max_output_tokens,
        buffer_tokens=100,
    )
    logger.info("over_context prompt (expect False) = %s", ok)

    over = gen._is_over_context_length(
        input=short_conversation,
        max_context_length=50,  # わざと小さく
        max_output_tokens=gen.max_output_tokens,
        buffer_tokens=100,
    )
    logger.info("over_context conversation (expect True) = %s", over)

    ok = gen._is_over_context_length(
        input=short_conversation,
        max_context_length=5000,
        max_output_tokens=gen.max_output_tokens,
        buffer_tokens=100,
    )
    logger.info("over_context conversation (expect False) = %s", ok)

    # -------------------------
    # C) chat: over-length 分岐と通常分岐
    # -------------------------
    logger.info("=== (C) chat ===")

    gen_small_ctx = OpenAIGenerator(
        model=model,
        max_context_length=1128,
        max_output_tokens=gen.max_output_tokens,
        logger=logger,
        api_key=api_key,
    )

    r_over = gen_small_ctx.chat(conversations=[short_conversation], buffer_tokens=1000)
    logger.info("chat(over) outputs[0].content = %r", r_over[0].outputs[0].content)

    r_ok = gen.chat(
        conversations=[short_conversation], buffer_tokens=10, temperature=0.2
    )
    logger.info("chat(ok) model_output = %r", r_ok[0].outputs[0].content)
    logger.info("chat(ok) metadata keys = %s", list((r_ok[0].metadata or {}).keys()))

    # -------------------------
    # D) completion
    # -------------------------
    logger.info("=== (D) completion ===")
    prompts = [
        Prompt.model_validate({"prompt": "Say 'OK' only."}),
        Prompt.model_validate({"prompt": "Give one haiku about winter."}),
    ]
    r_comp = gen.completion(prompts=prompts, temperature=0.2)
    for i, rr in enumerate(r_comp):
        logger.info("completion[%d] input=%r", i, rr.input)
        logger.info("completion[%d] output=%r", i, rr.outputs[0].content)
        if rr.metadata and "usage" in rr.metadata:
            logger.info("completion[%d] usage=%s", i, rr.metadata["usage"])


if __name__ == "__main__":
    load_dotenv()
    main()
