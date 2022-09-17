import random
from typing import Dict, List

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
        test_rate: float = 0.2,  # train, val
        shuffle: bool = False,
    ) -> None:
        self.seed = seed
        self.shuffle = shuffle
        self.dataset = load_dataset(dataset_path, dataset_name)
        self.prompt_input_dataset = self._convert_to_prompt_input_dataset(test_rate)

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

    def _convert_to_prompt_input_dataset(self, test_rate: float) -> Dict:
        every_dataset = self._get_preprocessed_dataset()

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(every_dataset)

        start_val_idx = int((1 - test_rate) * len(every_dataset))

        train_dataset, val_dataset = [], []
        for i, data in enumerate(every_dataset):
            input_example = InputExample(
                text_a=data["premise"],
                text_b=data["hypothesis"],
                label=data["label"],
                guid=data["guid"],
            )

            if i < start_val_idx:
                train_dataset.append(input_example)
            else:
                val_dataset.append(input_example)

        return {"train": train_dataset, "val": val_dataset}


class PromptLabeledDataModule:
    def __init__(self, dataset_path: str, seed: int = 42, test_rate: float = 0.2, shuffle: bool = False) -> None:
        self.seed = seed
        self.shuffle = shuffle
        self.dataset = self._load_labeled_dataset(dataset_path)
        self.prompt_input_dataset = self._convert_to_prompt_input_dataset(test_rate)

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

    def _convert_to_prompt_input_dataset(self, test_size: float = 0.2) -> Dict:
        answer_id_list = list(set((data.answer_id for data in self.dataset)))
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(answer_id_list)

        start_val_idx = int((1 - test_size) * len(answer_id_list))
        val_answer_id_set = set(answer_id_list[start_val_idx:])

        train_dataset, val_dataset = [], []
        for data in self.dataset:
            input_example = InputExample(
                text_a=data.premise,
                text_b=data.hypothesis,
                label=data.label,
                guid=data.guid,
            )

            if data.answer_id in val_answer_id_set:
                val_dataset.append(input_example)
            else:
                train_dataset.append(input_example)

        return {"train": train_dataset, "val": val_dataset}
