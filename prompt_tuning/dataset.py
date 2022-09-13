import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from datasets import load_dataset
from prompt_tuning.utils import log


class RequiredGradingData(BaseModel):
    problem_id: str  # Todo: 나중에 int로 바꾸기
    answer_id: int
    guid: str
    source: str
    premise: str
    hypothesis: str
    label: int
    criterion_idx: int


class PromptLoader:
    def __init__(
        self,
        max_seq_length: int,
        decoder_max_length: int,
        batch_size: int,
        teacher_forcing: bool,
        predict_eos_token: bool,
        truncate_method: str,
    ):
        self.max_seq_length = max_seq_length
        self.decoder_max_length = decoder_max_length
        self.batch_size = batch_size
        self.teacher_forcing = teacher_forcing
        self.predict_eos_token = predict_eos_token
        self.truncate_method = truncate_method

    def get_loader(
        self, dataset: List, template: ManualTemplate, tokenizer: PreTrainedTokenizer, tokenizer_wrapper_class
    ) -> PromptDataLoader:
        return PromptDataLoader(
            dataset=dataset,
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=tokenizer_wrapper_class,
            max_seq_length=self.max_seq_length,
            decoder_max_length=self.decoder_max_length,
            batch_size=self.batch_size,
            teacher_forcing=self.teacher_forcing,
            predict_eos_token=self.predict_eos_token,
            truncate_method=self.truncate_method,
        )


class PromptNliDataModule:
    def __init__(
        self,
        dataset_path: str = "klue",
        dataset_name: str = "nli",
        seed: int = 42,
        split_rate: Tuple = (8, 1, 1),  # train, val, test
        shuffle: bool = False,
    ) -> None:
        self.seed = seed
        self.shuffle = shuffle
        self.dataset = load_dataset(dataset_path, dataset_name)
        self.prompt_input_dataset = self._convert_to_prompt_input_dataset(split_rate)

    def _convert_to_prompt_input_dataset(self, split_rate: tuple) -> Dict:
        every_dataset = list(self.dataset["train"]) + list(self.dataset["validation"])

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(every_dataset)

        split_rate_sum = sum(split_rate)
        train_rate = split_rate[0] / split_rate_sum
        val_rate = split_rate[1] / split_rate_sum
        start_val_idx = int(train_rate * len(every_dataset))
        start_test_idx = int((train_rate + val_rate) * len(every_dataset))

        train_dataset, val_dataset, test_dataset = [], [], []
        for i, data in enumerate(every_dataset):
            input_example = InputExample(
                text_a=data["premise"],
                text_b=data["hypothesis"],
                label=data["label"],
                guid=data["guid"],
            )

            if i < start_val_idx:
                train_dataset.append(input_example)
            elif i < start_test_idx:
                val_dataset.append(input_example)
            else:
                test_dataset.append(input_example)

        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


class PromptLabeledDataModule:
    def __init__(
        self,
        dataset_path: str,
        seed: int = 42,
        split_rate: Tuple = (8, 1, 1),  # train, val, test
        shuffle: bool = False,
    ) -> None:
        self.seed = seed
        self.shuffle = shuffle
        self.dataset = self._load_labeled_dataset(dataset_path)
        self.prompt_input_dataset = self._split_dataset(split_rate)

    @staticmethod
    def _load_labeled_dataset(csv_path: str = "../static/labeled_dataset.csv") -> List[RequiredGradingData]:
        df = pd.read_csv(csv_path)
        log.info(f"previous data size : {len(df)}")
        df["user_answer"] = df["user_answer"].apply(
            lambda x: x.strip().replace("\n", "").replace("\xa0", "").replace("  ", " ")
        )
        df["user_answer"].replace("", np.nan, inplace=True)
        df.dropna(axis=0, subset=["user_answer"], inplace=True)  # 빈 답변 제거
        log.info(f"after data size : {len(df)}")

        if isinstance(df["correct_scoring_criterion"][0], str):  # list를 string으로 표현된 경우 type casting
            df["correct_scoring_criterion"] = df["correct_scoring_criterion"].apply(eval)
        if isinstance(df["scoring_criterion"][0], str):
            df["scoring_criterion"] = df["scoring_criterion"].apply(eval)

        dataset = []
        for row_data in df.itertuples():
            for i, criterion in enumerate(row_data.scoring_criterion):
                data = RequiredGradingData(
                    problem_id=row_data.problem_id,
                    answer_id=row_data.Index,
                    guid=f"{row_data.Index}-{i}",
                    source="CS-broker",
                    premise=row_data.user_answer,
                    hypothesis=criterion,
                    label=1 if criterion in row_data.correct_scoring_criterion else 0,
                    criterion_idx=i,
                )
                dataset.append(data)
        return dataset

    def _split_dataset(self, split_rate: Tuple = (8, 1, 1)) -> Dict:
        answer_id_list = list(set((data.answer_id for data in self.dataset)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(answer_id_list)
        split_rate_total = sum(split_rate)
        start_val_idx = int(split_rate[0] / split_rate_total * len(answer_id_list))
        start_test_idx = start_val_idx + int(split_rate[1] / split_rate_total * len(answer_id_list))
        train_answer_id_set = set(answer_id_list[:start_val_idx])
        val_answer_id_set = set(answer_id_list[start_val_idx:start_test_idx])
        test_answer_id_set = set(answer_id_list[start_test_idx:])

        assert len(train_answer_id_set) + len(val_answer_id_set) + len(test_answer_id_set) == len(
            answer_id_list
        ), f"dataset split error: split된 데이터와 전체 데이터의 갯수가 맞지 않습니다. {__file__}을 확인해주세요"

        train_dataset, val_dataset, test_dataset = [], [], []

        for data in self.dataset:
            if data.answer_id in train_answer_id_set:
                train_dataset.append(data)
            elif data.answer_id in val_answer_id_set:
                val_dataset.append(data)
            elif data.answer_id in test_answer_id_set:
                test_dataset.append(data)
            else:
                assert f"split error: dataset split 도중 에러가 발생하였습니다. {__file__}을 확인해주세요"

        return {"train": train_dataset, "val": val_dataset, "test": test_dataset}
