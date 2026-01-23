import json
import random
import re
import uuid
from logging import Logger
from typing import Any

import nltk
import wonderwords
from nltk.tokenize import sent_tokenize

from ..base import BaseDatasetGenerator
from ..config import NIAHSynthesisConfig
from ..data_model import Content, NIAHDatasetSchema

nltk.download("punkt")
nltk.download("punkt_tab")


class NIAHDatasetGenerator(
    BaseDatasetGenerator[NIAHDatasetSchema, NIAHSynthesisConfig]
):
    """
    Generates a dataset for the Needle-in-a-Haystack (NIAH) task by replicating
    the logic from the original RULER script.
    """

    @property
    def SCHEMA_CLASS(self) -> type[NIAHDatasetSchema]:
        return NIAHDatasetSchema

    def __init__(
        self, config: NIAHSynthesisConfig, logger: Logger | None = None
    ) -> None:
        super().__init__(config, logger)

        # Haystackと単語リストを初期化時に読み込む
        self._initialize_haystack()
        self._initialize_words()

        # config の num_needle_k が num_needle_q より小さい場合、合わせる
        self.config.num_needle_k = max(
            self.config.num_needle_k, self.config.num_needle_q
        )

    def _initialize_haystack(self) -> None:
        """Loads and prepares the haystack content based on the config."""
        self.haystack_source: list[str]
        if self.config.type_haystack == "essay":
            with self.config.paulgraham_essay_path.open(encoding="utf-8") as f:
                essay_text = json.load(f)["text"]
            self.haystack_source = re.sub(r"\s+", " ", essay_text).split(" ")
        elif self.config.type_haystack == "noise":
            self.haystack_source = [
                "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
            ]
        elif self.config.type_haystack == "needle":
            self.haystack_source = [self.config.needle_format]
        else:
            raise NotImplementedError(
                f"{self.config.type_haystack} is not implemented."
            )

    def _initialize_words(self) -> None:
        """Initializes the word list for generating random word-based needles."""
        try:
            nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
            adjs = wonderwords.random_word._get_words_from_text_file(
                "adjectivelist.txt"
            )

            self.words = sorted(
                list(set([f"{adj}-{noun}" for adj in adjs for noun in nouns]))
            )
        except FileNotFoundError:
            self.logger.warning(
                "wonderwords wordlists not found. You may need to run wonderwords once "
                "from the command line to download them. Using a small fallback list."
            )
            self.words = [
                "happy-cat",
                "blue-dog",
                "green-bird",
                "fast-car",
                "shiny-star",
            ]

    def _generate_random_value(self, value_type: str) -> str:
        """Generates a single random value of a specified type."""
        if value_type == "numbers":
            lower_bound = 10 ** (7 - 1)
            upper_bound = 10**7 - 1
            return str(random.randint(lower_bound, upper_bound))
        elif value_type == "words":
            return random.choice(self.words)
        elif value_type == "uuids":
            return str(uuid.UUID(int=random.getrandbits(128), version=4))
        else:
            raise NotImplementedError(f'Value type "{value_type}" is not implemented.')

    def _gen_one_sample(
        self, sample_index: int, num_units: int, **kwargs
    ) -> tuple[Content, dict[str, Any]]:
        """
        Generates a single sample for the NIAH task.
        """
        # 1. Needlesを生成
        keys: list[str] = []
        values: list[list[str]] = []
        needles: list[str] = []
        for _ in range(self.config.num_needle_k):
            key = self._generate_random_value(self.config.type_needle_k)
            keys.append(key)

            value_list = []
            for _ in range(self.config.num_needle_v):
                value = self._generate_random_value(self.config.type_needle_v)
                value_list.append(value)
                needles.append(
                    self.config.needle_format.format(
                        type_needle_v=self.config.type_needle_v, key=key, value=value
                    )
                )
            values.append(value_list)

        random.shuffle(needles)

        # 2. Context (Haystack + Needles) を生成
        if self.config.type_haystack == "essay":
            # num_units に応じて haystack の長さを調整
            source_len = len(self.haystack_source)
            repeats = (num_units + source_len - 1) // source_len
            text = " ".join((self.haystack_source * repeats)[:num_units])

            sents = sent_tokenize(text.strip())
            num_sents = len(sents)

            # 複数の深度をランダムにサンプリングして挿入位置を決定
            depths = list(range(0, 101, 5))
            insertion_depths = sorted(random.sample(depths, len(needles)))
            insertion_indices = [int(num_sents * (d / 100.0)) for d in insertion_depths]

            # 順番に挿入していく
            for i, idx in enumerate(insertion_indices):
                sents.insert(
                    idx + i, needles[i]
                )  # 挿入するたびにインデックスがずれるので+i
            context = " ".join(sents)

        else:  # 'noise' or 'needle'
            base_sentence = self.haystack_source[0]

            if self.config.type_haystack == "noise":
                sentences = [base_sentence] * num_units
            else:  # 'needle'
                sentences = [
                    base_sentence.format(
                        type_needle_v=self.config.type_needle_v,
                        key=self._generate_random_value(self.config.type_needle_k),
                        value=self._generate_random_value(self.config.type_needle_v),
                    )
                    for _ in range(num_units)
                ]
            # needleを挿入するインデックスを決定
            insertion_indices = sorted(random.sample(range(num_units), len(needles)))

            # 挿入インデックスから深度(%)を計算
            # 全体の長さは num_units なので、(インデックス / 全体の長さ) * 100 で算出
            if num_units > 0:
                insertion_depths = [
                    (idx / num_units) * 100.0 for idx in insertion_indices
                ]
            else:
                insertion_depths = []

            # 実際に文リストに挿入（逆順に処理しないとインデックスがずれる）
            temp_needles = list(needles)
            for index in sorted(insertion_indices, reverse=True):
                if temp_needles:
                    sentences.insert(index, temp_needles.pop())

            context = "\n".join(sentences)

        # 3. Query と Answer を生成
        query_indices = random.sample(
            range(self.config.num_needle_k), self.config.num_needle_q
        )
        query_keys = [keys[i] for i in query_indices]
        answers: list[str] = [val for i in query_indices for val in values[i]]

        if len(query_keys) > 1:
            query_str = ", ".join(query_keys[:-1]) + ", and " + query_keys[-1]
        else:
            query_str = query_keys[0]

        # 4. プロンプト全体を組み立て
        task_prompt_template = self.config.task_prompt_template
        answer_prefix_template = self.config.answer_prefix_template
        type_needle_v = self.config.type_needle_v

        # 単数形・複数形の調整
        if self.config.num_needle_q * self.config.num_needle_v == 1:
            task_prompt_template = task_prompt_template.replace(
                "Some special magic", "A special magic"
            )
            task_prompt_template = task_prompt_template.replace("are all", "is")
            task_prompt_template = task_prompt_template.replace("are", "is")
            answer_prefix_template = answer_prefix_template.replace("are", "is")
            if type_needle_v.endswith("s"):
                type_needle_v = type_needle_v[:-1]

        user_prompt = task_prompt_template.format(
            type_needle_v=type_needle_v,
            context=context,
            query=query_str,
        )
        answer_prefix = answer_prefix_template.format(
            type_needle_v=type_needle_v, query=query_str
        )

        content = Content(
            user_prompt=user_prompt, answer_prefix=answer_prefix, outputs=answers
        )
        extra_fields = {
            "needles": needles,
            "query": query_str,
            "answer_prefix": answer_prefix,
            "target_depth_percent": insertion_depths,
        }

        return content, extra_fields
