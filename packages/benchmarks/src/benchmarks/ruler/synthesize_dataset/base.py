import random
from abc import ABC, abstractmethod
from logging import Logger
from typing import Any, Generic, Sequence, Type, TypeVar

import numpy as np
from transformers import AutoTokenizer

from .config import BaseSynthesisConfig
from .data_model import BaseDatasetSchema, Content


ConfigType = TypeVar("ConfigType", bound=BaseSynthesisConfig)
SchemaType = TypeVar("SchemaType", bound=BaseDatasetSchema)


class BaseDatasetGenerator(ABC, Generic[SchemaType, ConfigType]):
    """
    データセットジェネレータの抽象基底クラス。（ドキュメントは同様）
    """

    @property
    @abstractmethod
    def SCHEMA_CLASS(self) -> Type[SchemaType]:
        """データセットの各行に対応するPydanticスキーマクラス。"""
        raise NotImplementedError

    def __init__(self, config: ConfigType, logger: Logger) -> None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        self.config: ConfigType = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.hf_tokenizer_path, trust_remote_code=True
        )
        self.logger: Logger = logger

    def _prompt_tokens(
        self, user_prompt: str, answer_prefix: str, with_chat_template: bool = True
    ) -> int:
        """
        チャット形式のプロンプトが消費する合計トークン数を返します。

        このカウントには、モデル固有のチャットテンプレートによって追加される
        全ての特殊トークンが含まれます。

        Parameters
        ----------
        prompt : str
            ユーザーに表示されるプロンプト文字列。

        Returns
        -------
        int
            テンプレート適用後のプロンプトが消費するトークン数。
        """

        if with_chat_template:
            prompt: str = (
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.config.apply_chat_template_kwargs,
                )
                + answer_prefix
            )
        else:
            prompt = user_prompt + answer_prefix
        ids = self.tokenizer.encode(prompt)
        return len(ids)

    def _within_context_limit(
        self, prompt_tokens: int, max_context_length: int
    ) -> bool:
        """
        プロンプトと予約された生成トークンがコンテキスト長に収まるかを判定します。

        Parameters
        ----------
        prompt_tokens : int
            テンプレート適用後プロンプトのトークン数。
        max_context_length : int
            モデルの最大コンテキストウィンドウ。

        Returns
        -------
        bool
            コンテキスト長に収まる場合は True。
        """
        return prompt_tokens + self.config.max_new_tokens <= max_context_length

    def _sample_total_tokens(self, num_units: int, **kwargs) -> tuple[int, Content]:
        """仮のサンプルを1つ生成し、そのプロンプトのトークン長を計測します。"""
        content, _ = self._gen_one_sample(sample_index=0, num_units=num_units, **kwargs)
        return self._prompt_tokens(
            user_prompt=content.user_prompt, answer_prefix=content.answer_prefix
        ), content

    @abstractmethod
    def _gen_one_sample(
        self, sample_index: int, num_units: int, **kwargs
    ) -> tuple[Content, dict[str, Any]]:
        """
        [要実装] サンプルを1つ生成します。

        サブクラスはこのメソッドをオーバーライドして、タスク固有のサンプル生成ロジックを
        実装する必要があります。
        Note: このメソッド内で、挿入深度をランダムに決定し、
        `target_depth_percent` キー でextra_fieldsに含めて返す必要があります。

        Returns
        -------
        tuple[Content, dict[str, Any]]
            - `Content`: `input`と`output`を含むオブジェクト。
            - `dict`: `needle`や`question`など、タスク固有の追加フィールドを含む辞書。
        """
        raise NotImplementedError

    def _optimal_units(self, max_context_length: int, **kwargs) -> int:
        """指定されたコンテキストウィンドウ内で許可される最大の `num_units` を見つけます。"""
        if max_context_length <= self.config.max_new_tokens:
            return 1

        low, high = 1, self.config.initial_units
        while True:
            tokens, _ = self._sample_total_tokens(high, **kwargs)
            if not self._within_context_limit(tokens, max_context_length):
                break
            low, high = high, high * 2
            if high > 1000000:
                break

        best = low
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                break
            tokens, _ = self._sample_total_tokens(mid, **kwargs)
            if self._within_context_limit(tokens, max_context_length):
                best, low = mid, mid + 1
            else:
                high = mid - 1
        return best

    def _generate_samples(self, max_context_length: int, **kwargs) -> list[SchemaType]:
        """指定されたコンテキストサイズでサンプルのバッチを生成し、検証します。"""
        max_units = self._optimal_units(max_context_length=max_context_length, **kwargs)

        results: list[SchemaType] = []
        limit = self.config.num_samples * self.config.retry_limit
        try_count = 0

        order = list(range(self.config.num_samples))
        random.shuffle(order)

        while len(results) < self.config.num_samples and try_count < limit:
            try_count += 1
            logical_idx = order[len(results)]
            for _num_units in range(max_units, self.config.min_units - 1, -1):
                content, extra_fields = self._gen_one_sample(
                    sample_index=logical_idx, num_units=_num_units, **kwargs
                )

                whole_text: str = content.user_prompt + content.answer_prefix
                if self.config.remove_newline_tab:
                    whole_text = " ".join(
                        whole_text.replace("\n", " ").replace("\t", " ").strip().split()
                    )
                answer_prefix_index = whole_text.rfind(content.answer_prefix[:10])
                content.answer_prefix = whole_text[answer_prefix_index:]
                content.user_prompt = whole_text[:answer_prefix_index]

                prompt_tokens = self._prompt_tokens(
                    user_prompt=content.user_prompt, answer_prefix=content.answer_prefix
                )

                # コンテキスト長に収まれば成功
                if self._within_context_limit(prompt_tokens, max_context_length):
                    content_length = self._prompt_tokens(
                        user_prompt=content.user_prompt,
                        answer_prefix=content.answer_prefix,
                        with_chat_template=False,
                    )

                    target_depth = extra_fields.pop("target_depth_percent", -1.0)
                    schema_data = {
                        "content": content,
                        "task": self.config.task,
                        "subset": self.config.subset,
                        "sample_id": len(results),
                        "input_content_length": content_length,
                        "input_length_w_model_temp": prompt_tokens,
                        "target_context_length": max_context_length,
                        "target_depth_percent": target_depth,
                        **extra_fields,
                    }
                    results.append(self.SCHEMA_CLASS(**schema_data))
                    break

        if len(results) < self.config.num_samples:
            raise RuntimeError(
                "試行回数の上限に達しましたが、要求されたサンプル数を生成できませんでした。 %s / %s",
                len(results),
                self.config.num_samples,
            )
        return results

    def _save_dataset(self, samples: Sequence[SchemaType]) -> None:
        path = (
            self.config.save_dirpath / self.config.task / f"{self.config.subset}.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            for rec in samples:
                fp.write(rec.model_dump_json() + "\n")
        self.logger.info("データセットを %s に保存しました。", path)

    def run(self, **kwargs) -> None:
        self.logger.info(
            "タスク %s/%s のデータセット生成を開始します。",
            self.config.task,
            self.config.subset,
        )
        all_samples: list[SchemaType] = []
        for ctx_len in sorted(self.config.context_lengths):
            self.logger.info(
                "コンテキスト長: %s のサンプルを %s 個生成中...",
                ctx_len,
                self.config.num_samples,
            )
            all_samples.extend(
                self._generate_samples(max_context_length=ctx_len, **kwargs)
            )

        self._save_dataset(all_samples)
        self.logger.info("データセットの生成が完了しました。")
