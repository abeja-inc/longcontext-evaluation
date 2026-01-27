import dataclasses
from logging import Logger
from typing import Any

from sglang import Engine
from sglang.srt.server_args import ServerArgs

from .base import BaseGenerator
from .data import Conversation, OutputContent, Prompt, Response
from .reasoning_parser import BaseReasoningParser, resolve_reasoning_parser


class SGLangOfflineGenerator(BaseGenerator):
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
        server_args = ServerArgs(
            model_path=self.model_name, context_length=self.max_context_length, **kwargs
        )
        self.llm = Engine(**dataclasses.asdict(server_args))
        self.tokenizer = self.llm.tokenizer_manager.tokenizer
        self.reasoning_parser = resolve_reasoning_parser(reasoning_parser)

    def _count_tokens(self, input: Prompt | Conversation, **kwargs: Any) -> int:
        if isinstance(input, Conversation):
            return len(
                self.tokenizer.apply_chat_template(
                    input.prompt, tokenize=True, **kwargs
                )
            )
        else:
            return len(self.tokenizer.encode(input.prompt, add_special_tokens=False))

    def _format_response(
        self,
        inputs: list[Prompt] | list[Conversation],
        sglang_responses: list[dict[str, Any]],
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
                sglang_resp = sglang_responses[out_pos]
                out_pos += 1
                response_outputs: list[OutputContent]
                if self.reasoning_parser:
                    reasoning_content, content = self.reasoning_parser.parse(
                        sglang_resp["text"]
                    )
                    response_outputs = [
                        OutputContent(
                            content=content, reasoning_content=reasoning_content
                        )
                    ]
                else:
                    response_outputs = [OutputContent(content=sglang_resp["text"])]

                response = Response(
                    input=input.prompt,
                    outputs=response_outputs,
                    metadata=input.metadata,
                )
            formatted_responses.append(response)
        return formatted_responses

    def _generate(
        self, prompts: list[str], sampling_params: dict[str, Any], **kwargs: Any
    ) -> list[dict[str, Any]]:
        if "n" in sampling_params and sampling_params["n"] > 1:
            raise ValueError(
                "n > 1 is not supported in sglang offline inference. "
                "The official SGLang documentation discourages multiple outputs per request. "
                "Use n=1 and duplicate prompts in the batch instead.\n"
                "Sampling Parameters in SGLang: https://docs.sglang.io/basic_usage/sampling_params.html"
            )

        outputs: list[dict[str, Any]] = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            **kwargs,
        )
        return outputs

    def chat(
        self,
        *,
        conversations: list[Conversation],
        sampling_params: dict[str, Any] = {},
        buffer_tokens: int = 10,
        chat_template_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> list[Response]:
        sampling_params["max_tokens"] = self.max_output_tokens
        self.logger.info(f"Sampling parameters: {sampling_params}")

        chat_template_kwargs["tokenize"] = False
        chat_template_kwargs["add_generation_prompt"] = True

        filtered_conversations, skip_idx = self._filter_long_inputs(
            inputs=conversations,
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens,
            buffer_tokens=buffer_tokens,
            **chat_template_kwargs,
        )
        responses: list[dict[str, Any]] = self._generate(
            [
                self.tokenizer.apply_chat_template(
                    conversation.prompt,
                    **chat_template_kwargs,
                )
                for conversation in filtered_conversations
            ],
            sampling_params=sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            **kwargs,
        )
        return self._format_response(
            inputs=conversations, sglang_responses=responses, skip_idx=skip_idx
        )

    def completion(
        self,
        *,
        prompts: list[Prompt],
        sampling_params: dict[str, Any] = {},
        buffer_tokens: int = 10,
        **kwargs: Any,
    ) -> list[Response]:
        sampling_params["max_tokens"] = self.max_output_tokens
        self.logger.info(f"Sampling parameters: {sampling_params}")

        filtered_prompts, skip_idx = self._filter_long_inputs(
            inputs=prompts,
            max_context_length=self.max_context_length,
            max_output_tokens=self.max_output_tokens,
            buffer_tokens=buffer_tokens,
        )
        responses: list[dict[str, Any]] = self._generate(
            [prompt.prompt for prompt in filtered_prompts],
            sampling_params=sampling_params,
            **kwargs,
        )
        return self._format_response(
            inputs=prompts, sglang_responses=sglang_responses, skip_idx=skip_idx
        )
