from datasets import load_dataset


def get_klue_dataset(task: str):
    return load_dataset("klue", task)
