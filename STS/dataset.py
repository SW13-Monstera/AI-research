from datasets import load_dataset


def get_klue_dataset(task: str):
    dataset = load_dataset('klue', task)
    dataset = dataset.flatten()
    dataset = dataset.rename_column('labels.real-label', 'label')
    return dataset.remove_columns(['labels.label', 'labels.binary-label'])
