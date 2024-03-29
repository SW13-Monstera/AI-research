import random
from typing import List

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

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
        self.max_seq_length: int = max_seq_length
        self.decoder_max_length: int = decoder_max_length
        self.batch_size: int = batch_size
        self.teacher_forcing: bool = teacher_forcing
        self.predict_eos_token: bool = predict_eos_token
        self.truncate_method: str = truncate_method

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
        shuffle: bool = False,
    ) -> None:
        self.seed: int = seed
        self.shuffle: bool = shuffle
        self.dataset: DatasetDict = load_dataset(dataset_path, dataset_name)
        self.prompt_input_dataset: List[InputExample] = self._convert_to_prompt_input_dataset()

    def _get_preprocessed_dataset(self) -> List:
        """
        [before] 0: entailment, 1: neutral, 2: contradiction
        우리의 task에 맞게 변형하려면 neutral을 삭제하고 entailment과 contradiction를 swap해야 한다.
        [after] 0: contradiction, 1: entailment
        """
        new_dataset = []
        for data_type in ["train", "validation"]:
            for data in self.dataset[data_type]:
                if data["label"] == 1:
                    continue
                elif data["label"] == 2:
                    data["label"] = 0
                elif data["label"] == 0:
                    data["label"] = 1
                new_dataset.append(data)
        return new_dataset

    def _convert_to_prompt_input_dataset(self) -> List[InputExample]:
        every_dataset = self._get_preprocessed_dataset()

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(every_dataset)

        train_dataset = []
        for data in every_dataset:
            input_example = InputExample(
                text_a=data["premise"],
                text_b=data["hypothesis"],
                label=data["label"],
                guid=data["guid"],
            )
            train_dataset.append(input_example)

        return train_dataset


class PromptLabeledDataModule:
    def __init__(self, dataset_path: str, seed: int = 42, shuffle: bool = False) -> None:
        self.seed: int = seed
        self.shuffle: bool = shuffle
        self.dataset: List[RequiredGradingData] = self._load_labeled_dataset(dataset_path)
        self.prompt_input_dataset: List[InputExample] = self._convert_to_prompt_input_dataset()

    @staticmethod
    def _load_labeled_dataset(csv_path: str) -> List[RequiredGradingData]:
        df = pd.read_csv(csv_path)
        log.info(f"previous data size : {len(df)}")
        df.dropna(axis=0, subset=["user_answer"], inplace=True)  # 빈 답변 제거
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

    def _convert_to_prompt_input_dataset(self) -> List[InputExample]:
        train_dataset = []
        for data in self.dataset:
            input_example = InputExample(
                text_a=data.premise,
                text_b=data.hypothesis,
                label=data.label,
                guid=data.answer_id,
            )
            train_dataset.append(input_example)
        return train_dataset
