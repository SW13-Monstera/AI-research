import os

import torch
from dataset import get_klue_dataset
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from utils import seed_everything


def preprocess_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length=512,
        padding=True,
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]
    eval_result = metric.compute(predictions=predictions, references=labels)
    return eval_result


if __name__ == "__main__":

    seed = 42
    num_labels = 1
    model_name = "klue/bert-base"
    learning_rate = 2e-5
    epoch = 30
    weight_decay = 0.01
    warmup_steps = 200
    metric_name = "pearsonr"
    batch_size = 64

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_klue_dataset("sts")
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    output_dir = os.path.join("test-klue", "sts")
    logging_dir = os.path.join(output_dir, "logs")

    training_args = TrainingArguments(
        # checkpoint
        output_dir=output_dir,
        # overwrite_output_dir=True,
        # Model Save & Load
        save_strategy="epoch",  # 'steps'
        load_best_model_at_end=True,
        # save_steps = 500,
        # Dataset
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # Optimizer
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        # warmup_steps=warmup_steps,
        # Resularization
        # max_grad_norm=1.0,
        # label_smoothing_factor=0.1,
        # Evaluation
        metric_for_best_model="eval_" + metric_name,
        evaluation_strategy="epoch",
        # Huggingface Hub Upload
        # push_to_hub=True,
        # push_to_hub_model_id=f"{model_name}-finetuned-{task}",
        # Logging
        logging_dir=logging_dir,
        # report_to="wandb",
        # Randomness
        seed=42,
    )
    metric = load_metric(metric_name)
    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
