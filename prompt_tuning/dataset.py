import random
from typing import Tuple

from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate
from transformers import PreTrainedTokenizer

from datasets import load_dataset


class PromptDataModule:
    def __init__(
        self,
        dataset_path: str = "klue",
        dataset_name: str = "nli",
        seed: int = 42,
        split_rate: tuple = (8, 1, 1),  # train, val, test
        max_seq_length: int = 512,
        decoder_max_length: int = 3,
        batch_size: int = 4,
        teacher_forcing: bool = False,
        predict_eos_token: bool = False,
        truncate_method: str = "head",
        shuffle: bool = False,
    ) -> None:
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.decoder_max_length = decoder_max_length
        self.batch_size = batch_size
        self.teacher_forcing = teacher_forcing
        self.predict_eos_token = predict_eos_token
        self.truncate_method = truncate_method
        self.shuffle = shuffle
        self.dataset = load_dataset(dataset_path, dataset_name)
        self.prompt_input_dataset = self._convert_to_prompt_input_dataset(split_rate)

    def _convert_to_prompt_input_dataset(self, split_rate: tuple) -> dict:
        prompt_input_dataset = {"train": [], "val": [], "test": []}
        every_dataset = list(self.dataset["train"]) + list(self.dataset["validation"])

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(every_dataset)

        split_rate_sum = sum(split_rate)
        train_rate = split_rate[0] / split_rate_sum
        val_rate = split_rate[1] / split_rate_sum
        start_val_idx = int(train_rate * len(every_dataset))
        start_test_idx = int((train_rate + val_rate) * len(every_dataset))
        for i, data in enumerate(every_dataset):
            input_example = InputExample(
                text_a=data["premise"],
                text_b=data["hypothesis"],
                label=data["label"],
                guid=data["guid"],
            )

            if i < start_val_idx:
                prompt_input_dataset["train"].append(input_example)
            elif i < start_test_idx:
                prompt_input_dataset["val"].append(input_example)
            else:
                prompt_input_dataset["test"].append(input_example)

        return prompt_input_dataset

    def get_data_loader(
        self,
        template: ManualTemplate,
        tokenizer: PreTrainedTokenizer,
        tokenizer_wrapper_class,
    ) -> Tuple[PromptDataLoader, PromptDataLoader, PromptDataLoader]:
        train_data_loader, val_data_loader, test_data_loader = [
            PromptDataLoader(
                dataset=self.prompt_input_dataset[data_type],
                template=template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=tokenizer_wrapper_class,
                max_seq_length=self.max_seq_length,
                decoder_max_length=self.decoder_max_length,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                teacher_forcing=self.teacher_forcing,
                predict_eos_token=self.predict_eos_token,
                truncate_method=self.truncate_method,
            )
            for data_type in ["train", "val", "test"]
        ]
        return train_data_loader, val_data_loader, test_data_loader
