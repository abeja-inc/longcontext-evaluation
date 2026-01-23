import logging

from llm_inference.data import Conversation, Prompt
from llm_inference.vllm_offline_inference import VLLMOfflineGenerator
from vllm import SamplingParams


# -----------------------------
# 1) logger
# -----------------------------
def build_logger() -> logging.Logger:
    logger = logging.getLogger("smoke_test_vllm_generator")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def main() -> None:
    generator_config = {
        "dtype": "float32",
        "tensor_parallel_size": 8,
        "trust_remote_code": True,
        "max_num_seqs": 256,
        "hf_overrides": {
            "rope_parameters": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "truncate": False
            }
        },
        "quantization": None,
    }
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.95,
    )

    chat_template_kwargs = {"reasoning_effort": "high"}

    logger = build_logger()

    model = "/workspace/models/gpt-oss-20b"
    reasoning_parser = "openai_gptoss"
    max_output_tokens = 2048
    max_context_length = 3072

    gen = VLLMOfflineGenerator(
        model_name=model,
        max_context_length=max_context_length,
        max_output_tokens=max_output_tokens,
        logger=logger,
        reasoning_parser=reasoning_parser,
        **generator_config,
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

    tok3 = gen._count_tokens(input=short_conversation, **chat_template_kwargs)
    logger.info("tokens(conv enable thinking) = %s", tok3)

    # -------------------------
    # B) _is_over_context_length
    # -------------------------
    logger.info("=== (B) _is_over_context_length ===")

    over = gen._is_over_context_length(
        input=short_prompt,
        max_context_length=50,  # わざと小さく
        max_output_tokens=max_output_tokens,
        buffer_tokens=100,
    )
    logger.info("over_context prompt (expect True) = %s", over)

    ok = gen._is_over_context_length(
        input=short_prompt,
        max_context_length=5000,
        max_output_tokens=max_output_tokens,
        buffer_tokens=100,
    )
    logger.info("over_context prompt (expect False) = %s", ok)

    over = gen._is_over_context_length(
        input=short_conversation,
        max_context_length=50,  # わざと小さく
        max_output_tokens=max_output_tokens,
        buffer_tokens=100,
        **chat_template_kwargs,
    )
    logger.info("over_context conversation (expect True) = %s", over)

    ok = gen._is_over_context_length(
        input=short_conversation,
        max_context_length=5000,
        max_output_tokens=max_output_tokens,
        buffer_tokens=100,
        **chat_template_kwargs,
    )
    logger.info("over_context conversation (expect False) = %s", ok)

    # -------------------------
    # C) chat: over-length 分岐と通常分岐
    # -------------------------
    logger.info("=== (C) chat ===")

    r_over = gen.chat(
        conversations=[short_conversation],
        buffer_tokens=1000,
        sampling_params=sampling_params,
        chat_template_kwargs=chat_template_kwargs,
    )
    logger.info("chat(over) outputs[0].content = %r", r_over[0].outputs[0].content)

    r_ok = gen.chat(
        conversations=[short_conversation],
        buffer_tokens=10,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )
    logger.info("chat(ok) model_output = %r", r_ok[0].outputs[0].content)
    logger.info("chat(ok) metadata keys = %s", list((r_ok[0].metadata or {}).keys()))

    r_ok = gen.chat(
        conversations=[short_conversation],
        buffer_tokens=10,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": True},
    )
    logger.info("chat reasoning(ok) model_output = %r", r_ok[0].outputs[0].content)
    logger.info(
        "chat reasoning(ok) model_output_reasoning = %r",
        r_ok[0].outputs[0].reasoning_content,
    )
    logger.info(
        "chat reasoning(ok) metadata keys = %s", list((r_ok[0].metadata or {}).keys())
    )

    # -------------------------
    # D) completion
    # -------------------------
    logger.info("=== (D) completion ===")
    prompts = [
        Prompt.model_validate({"prompt": "Say 'OK' only."}),
        Prompt.model_validate({"prompt": "Give one haiku about winter."}),
    ]
    r_comp = gen.completion(prompts=prompts, sampling_params=sampling_params)
    for i, rr in enumerate(r_comp):
        logger.info("completion[%d] input=%r", i, rr.input)
        logger.info("completion[%d] output=%r", i, rr.outputs[0].content)


if __name__ == "__main__":
    main()
