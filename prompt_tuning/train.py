import os

import hydra
import pyrootutils
import torch
import wandb
from loss import EarlyStopping, calculate_metric, get_predicts
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
    epochs: int,
    model: PromptForClassification,
    train_data_loader: PromptDataLoader,
    val_data_loader: PromptDataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    logging_steps: int,
) -> None:
    wandb.watch(model, criterion, log="all", log_freq=100)
    early_stopping = EarlyStopping(patience=3, verbose=True)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for step, inputs in enumerate(tqdm(train_data_loader)):
            predicts = get_predicts(inputs, model)
            labels = inputs.label.cpu().numpy()
            loss = criterion(labels, predicts)
            loss.backward()
            total_loss += loss.item()
            del inputs, loss
            optimizer.step()
            optimizer.zero_grad()
            if step % logging_steps == 1:
                print_result(test_type="train", epoch=epoch, step=step, loss=total_loss)
            torch.cuda.empty_cache()

        test(model, val_data_loader, criterion, early_stopping)

        if early_stopping.early_stop:
            log.info("Early stopping")
            break


def test(
    model: PromptForClassification,
    test_data_loader: PromptDataLoader,
    criterion: torch.nn.Module,
    early_stopping: EarlyStopping,
) -> None:
    total_loss = total_acc = total_f1 = total_auc = 0

    with torch.no_grad():
        for inputs in tqdm(test_data_loader):
            predicts = get_predicts(inputs, model)
            labels = inputs.label.cpu().numpy()
            loss = criterion(labels, predicts)
            acc, f1, auc = calculate_metric(labels, predicts)
            total_loss += loss.item()
            total_acc += acc
            total_f1 += f1
            total_auc += auc
            del inputs, loss
    avg_loss = total_loss / len(test_data_loader)
    early_stopping(avg_loss, model)

    print_result(
        test_type="val",
        step=len(test_data_loader),
        loss=total_loss,
        accuracy_score=total_acc,
        f1_score=total_f1,
        auc=total_auc,
    )
    torch.cuda.empty_cache()


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    wandb.init(project="CS-broker", entity="ekzm8523", config=cfg)
    seed_everything(cfg.seed)
    log.info(cfg)

    model_class = get_model_class(plm_type=cfg.model.name)
    plm = model_class.model.from_pretrained(cfg.model.path, use_auth_token=HUGGING_FACE_ACCESS_TOKEN)
    tokenizer = model_class.tokenizer.from_pretrained(cfg.model.path, use_auth_token=HUGGING_FACE_ACCESS_TOKEN)

    special_tokens = ["</s>", "<unk>", "<pad>"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    log.info(f"{num_added_tokens}개의 special token 생성")
    log.info(f"new special_token : {special_tokens}")

    WrapperClass = model_class.wrapper

    nli_data_module: PromptNliDataModule = hydra.utils.instantiate(cfg.dataset.nli)

    template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

    prompt_loader: PromptLoader = hydra.utils.instantiate(cfg.dataset.loader)
    train_data_loader, val_data_loader = [
        prompt_loader.get_loader(
            dataset=nli_data_module.prompt_input_dataset[data_type],
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
        )
        for data_type in ["train", "val"]
    ]
    verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=2, label_words=[["yes"], ["no"]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"running on : {device}")
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)  # freeze 고려
    prompt_model.to(device)
    # prompt_model.load_state_dict(torch.load("./jw-mt5-base.bin"))
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
        epochs=cfg.epochs,
        model=prompt_model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        criterion=criterion,
        optimizer=optimizer,
        logging_steps=cfg.logging_steps,
    )

    labeled_data_module: PromptLabeledDataModule = hydra.utils.instantiate(cfg.dataset.labeled)
    train_data_loader, val_data_loader = [
        prompt_loader.get_loader(
            dataset=labeled_data_module.prompt_input_dataset[data_type],
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
        )
        for data_type in ["train", "val"]
    ]

    train(
        epochs=cfg.epochs,
        model=prompt_model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        criterion=criterion,
        optimizer=optimizer,
        logging_steps=cfg.logging_steps,
    )

    if cfg.upload_model_to_s3:
        date_folder = sorted(os.listdir("./outputs"))[-1]
        time_folder = sorted(os.listdir(f"./outputs/{date_folder}"))[-1]
        folder = os.path.join("ai-models", date_folder, time_folder)
        upload_model_to_s3(
            model=prompt_model,
            bucket=cfg.s3_bucket,
            folder=folder,
            model_name="model.pth",
        )


if __name__ == "__main__":
    main()
