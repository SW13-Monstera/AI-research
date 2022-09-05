import random

from openprompt.data_utils import InputExample

from datasets import load_dataset


class PromptPreprocessor:
    def __init__(
        self,
        dataset_path: str = "klue",
        dataset_name: str = "nli",
        seed: int = 42,
        split_rate: tuple = (8, 1, 1),  # train, val, test
    ):
        self.dataset = load_dataset(dataset_path, dataset_name)
        self.seed = seed
        self.prompt_input_dataset = self.convert_to_prompt_input_dataset(split_rate)

    def convert_to_prompt_input_dataset(
        self, split_rate: tuple, shuffle: bool = False
    ) -> dict:
        prompt_input_dataset = {"train": [], "val": [], "test": []}
        every_dataset = list(self.dataset["train"]) + list(self.dataset["val"])

        if shuffle:
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
