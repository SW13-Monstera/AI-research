import os
from typing import Union

import hydra
import pyrootutils
import torch
import wandb
from loss import EarlyStopping, calculate_metric
from omegaconf import DictConfig
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import get_model_class
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from torch.optim import AdamW
from tqdm import tqdm
from utils import log, print_result, seed_everything, upload_model_to_s3

from core.config import HUGGING_FACE_ACCESS_TOKEN
from prompt_tuning.dataset import (
    PromptLabeledDataModule,
    PromptLoader,
    PromptNliDataModule,
)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


def train(
    cfg: DictConfig,
    model: PromptForClassification,
    train_data_loader: PromptDataLoader,
    val_data_loader: PromptDataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    # wandb.watch(model, criterion, log="all", log_freq=100)
    date_folder = sorted(os.listdir("./outputs"))[-1]
    time_folder = sorted(os.listdir(f"./outputs/{date_folder}"))[-1]
    early_stopping = EarlyStopping(
        patience=cfg.early_stopping, verbose=True, path=f"outputs/{date_folder}/{time_folder}/best_model.pt"
    )
    device = model.device
    for epoch in range(cfg.epochs):
        total_loss = 0
        model.train()
        for step, inputs in enumerate(tqdm(train_data_loader)):
            inputs = inputs.to(device)
            logits = model(inputs)
            loss = criterion(logits, inputs.label)
            loss.backward()
            total_loss += loss.item()
            del inputs, loss
            optimizer.step()
            optimizer.zero_grad()
            if step % cfg.logging_steps == 0 and step >= cfg.logging_steps:
                print_result(test_type="train", epoch=epoch, step=step + 1, loss=total_loss)
            torch.cuda.empty_cache()

        test(model, val_data_loader, criterion, early_stopping)

        if early_stopping.early_stop:
            log.info("Early stopping")
            log.info(f"best auuracy is {early_stopping.best_acc}")
            break


def test(
    model: PromptForClassification,
    test_data_loader: PromptDataLoader,
    criterion: torch.nn.Module,
    early_stopping: EarlyStopping,
) -> None:
    total_loss = total_acc = total_f1 = 0
    device = model.device
    with torch.no_grad():
        for inputs in tqdm(test_data_loader):
            inputs = inputs.to(device)
            logits = model(inputs)
            predicts = torch.argmax(logits, dim=1).cpu().numpy()
            loss = criterion(logits, inputs.label)
            acc, f1 = calculate_metric(inputs.label.cpu().numpy(), predicts)
            total_loss += loss.item()
            total_acc += acc
            total_f1 += f1
            del inputs, loss
    avg_loss = total_loss / len(test_data_loader)
    acc = total_acc / len(test_data_loader)
    early_stopping(avg_loss, model, acc)

    print_result(
        test_type="val",
        step=len(test_data_loader),
        loss=total_loss,
        accuracy_score=total_acc,
        f1_score=total_f1,
    )
    torch.cuda.empty_cache()


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    experiment_description = input("experiment description : ")
    seed_everything(cfg.seed)
    log.info(cfg)

    date_folder = sorted(os.listdir("./outputs"))[-1]
    time_folder = sorted(os.listdir(f"./outputs/{date_folder}"))[-1]
    wandb.init(
        project="CS-broker",
        entity="ekzm8523",
        config=cfg,
        name=f"{date_folder}-{time_folder}",
        notes=experiment_description,
    )

    model_class = get_model_class(plm_type=cfg.model.name)
    plm = model_class.model.from_pretrained(cfg.model.path, use_auth_token=HUGGING_FACE_ACCESS_TOKEN)
    tokenizer = model_class.tokenizer.from_pretrained(cfg.model.path, use_auth_token=HUGGING_FACE_ACCESS_TOKEN)

    special_tokens = ["</s>", "<unk>", "<pad>"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    log.info(f"{num_added_tokens}개의 special token 생성")
    log.info(f"new special_token : {special_tokens}")

    WrapperClass = model_class.wrapper
    template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

    prompt_loader: PromptLoader = hydra.utils.instantiate(cfg.dataset.loader)

    train_data_module: Union[PromptLabeledDataModule, PromptNliDataModule] = hydra.utils.instantiate(cfg.dataset.train)
    test_data_module: PromptLabeledDataModule = hydra.utils.instantiate(cfg.dataset.test)
    train_data_loader = prompt_loader.get_loader(
        dataset=train_data_module.prompt_input_dataset,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
    )
    test_data_loader = prompt_loader.get_loader(
        dataset=test_data_module.prompt_input_dataset,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
    )

    verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=2, label_words=[["yes"], ["no"]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"running on : {device}")
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)  # freeze 고려
    prompt_model.load_state_dict(torch.load("./outputs/2022-09-18/18-10-05/best_model.pt"))
    prompt_model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # loss 생각해보기
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)
    train(
        cfg=cfg,
        model=prompt_model,
        train_data_loader=train_data_loader,
        val_data_loader=test_data_loader,
        criterion=criterion,
        optimizer=optimizer,
    )

    local_model_path = f"outputs/{date_folder}/{time_folder}/best_model.pt"
    if cfg.upload_model_to_s3:
        folder = f"ai-models/{date_folder}/{time_folder}"
        upload_model_to_s3(
            local_path=local_model_path,
            bucket=cfg.s3_bucket,
            folder=folder,
            model_name="best_model.pt",
        )


if __name__ == "__main__":
    main()
