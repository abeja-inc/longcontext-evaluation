from logging import Logger
from typing import Any

import vllm
from packaging.version import parse as parse_version
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput as VLLMResponse

from .base import BaseGenerator
from .data import Conversation, OutputContent, Prompt, Response
from .reasoning_parser import BaseReasoningParser, resolve_reasoning_parser
from .token_counter import TokenCounter


class VLLMOfflineGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        max_context_length: int,
        max_output_tokens: int,
        logger: Logger,
        reasoning_parser: str | BaseReasoningParser | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_context_length=max_context_length,
            max_output_tokens=max_output_tokens,
            logger=logger,
        )

        self.llm = LLM(
            model=self.model_name, max_model_len=self.max_context_length, **kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.reasoning_parser = resolve_reasoning_parser(reasoning_parser)
        self.tokenizer_type = "huggingface"
        self.token_counter = TokenCounter.from_tokenizer(
            tokenizer=self.tokenizer, tokenizer_type=self.tokenizer_type
        )

    def _format_response(
        self,
        inputs: list[Prompt] | list[Conversation],
        vllm_responses: VLLMResponse,
        skip_idx: list[int],
    ) -> list[Response]:
        out_pos = 0
        formatted_responses: list[Response] = []
        for idx, input in enumerate(inputs):
            if idx in skip_idx:
                response = Response(
                    input=input.prompt,
                    outputs=[
                        OutputContent(content=self.default_too_long_input_error_message)
                    ],
                    metadata=input.metadata,
                )
            else:
                vllm_resp = vllm_responses[out_pos]
                out_pos += 1
                response_outputs: list[OutputContent]
                if self.reasoning_parser:
                    response_outputs = []
                    for output in vllm_resp.outputs:
                        reasoning_content, content = self.reasoning_parser.parse(
                            output.text
                        )
                        response_outputs.append(
                            OutputContent(
                                content=content, reasoning_content=reasoning_content
                            )
                        )
                else:
                    response_outputs = [
                        OutputContent(content=output.text)
                        for output in vllm_resp.outputs
                    ]

                response = Response(
                    input=input.prompt,
                    outputs=response_outputs,
                    metadata=input.metadata,
                )
            formatted_responses.append(response)
        return formatted_responses

    def _chat(
        self,
        *,
        conversations: list[Conversation],
        sampling_params: SamplingParams | dict[str, Any],
        buffer_tokens: int = 0,
        chat_template_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Response]:
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**sampling_params)
        if sampling_params:
            sampling_params.max_tokens = self.max_output_tokens
        else:
            sampling_params = SamplingParams(max_tokens=self.max_output_tokens)
        self.logger.info(f"Sampling parameters: {sampling_params}")

        filtered_conversations, skip_idx = self._filter_long_inputs(
            inputs=conversations,
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens,
            buffer_tokens=buffer_tokens,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=True,
        )

        vllm_version_raw = getattr(vllm, "__version__", None)
        use_completion_fallback = vllm_version_raw is not None and parse_version(
            vllm_version_raw
        ) <= parse_version("0.8.5")

        try:
            if use_completion_fallback:
                raise RuntimeError("Force completion mode for vLLM <= 0.8.5")

            responses: list[VLLMResponse] = self.llm.chat(
                [conversation.prompt for conversation in filtered_conversations],
                sampling_params=sampling_params,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            )
        except Exception as e:
            self.logger.warning(
                "Using completion fallback "
                f"(vLLM version={vllm_version_raw or 'unknown'}): {e}"
            )

            filtered_prompts = [
                Prompt(
                    prompt=self.tokenizer.apply_chat_template(
                        conversation.prompt,
                        **chat_template_kwargs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                for conversation in filtered_conversations
            ]

            responses: list[VLLMResponse] = self.llm.generate(
                [prompt.prompt for prompt in filtered_prompts],
                sampling_params=sampling_params,
                **kwargs,
            )
        return self._format_response(
            inputs=conversations, vllm_responses=responses, skip_idx=skip_idx
        )

    def _completion(
        self,
        *,
        prompts: list[Prompt],
        sampling_params: SamplingParams | None = None,
        buffer_tokens: int = 0,
        **kwargs: Any,
    ) -> list[Response]:
        if sampling_params:
            sampling_params.max_tokens = self.max_output_tokens
        else:
            sampling_params = SamplingParams(max_tokens=self.max_output_tokens)
        self.logger.info(f"Sampling parameters: {sampling_params}")

        filtered_prompts, skip_idx = self._filter_long_inputs(
            inputs=prompts,
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens,
            buffer_tokens=buffer_tokens,
        )
        responses: list[VLLMResponse] = self.llm.generate(
            [prompt.prompt for prompt in filtered_prompts],
            sampling_params=sampling_params,
            **kwargs,
        )
        return self._format_response(
            inputs=prompts, vllm_responses=responses, skip_idx=skip_idx
        )
