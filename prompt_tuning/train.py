import random

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm

from core.utils import seed_everything
from datasets import load_dataset

if __name__ == "__main__":
    seed = 42
    task = "nli"

    seed_everything(seed)
    dataset = load_dataset("klue", task)
    prompt_input_dataset = {"train": [], "validation": [], "test": []}
    for data in dataset["train"]:  # train data
        input_example = InputExample(
            text_a=data["premise"],
            text_b=data["hypothesis"],
            label=data["label"],
            guid=data["guid"],
        )
        prompt_input_dataset["train"].append(input_example)

    validation_dataset = list(dataset["validation"])
    random.seed(seed)
    random.shuffle(validation_dataset)

    for i, data in enumerate(validation_dataset):
        input_example = InputExample(
            text_a=data["premise"],
            text_b=data["hypothesis"],
            label=data["label"],
            guid=data["guid"],
        )
        if i % 2 == 0:
            prompt_input_dataset["validation"].append(input_example)
        else:
            prompt_input_dataset["test"].append(input_example)
    pretrain_model_path = "google/mt5-base"
    plm, tokenizer, model_config, WrapperClass = load_plm("t5", pretrain_model_path)
