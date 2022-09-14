import hydra
import pyrootutils
import torch
from loss import evaluation
from omegaconf import DictConfig
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from torch.optim import AdamW
from tqdm import tqdm
from utils import log, print_result, seed_everything

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
    best_acc = 0
    for epoch in range(epochs):
        total_loss = total_acc = total_f1 = 0
        model.train()
        for step, inputs in enumerate(tqdm(train_data_loader)):
            loss, acc, f1 = evaluation(inputs, model, criterion)
            del inputs
            loss.backward()
            total_loss += loss.item()
            total_acc += acc
            total_f1 += f1
            del loss
            optimizer.step()
            optimizer.zero_grad()
            if step % logging_steps == 1:
                print_result(
                    test_type="train",
                    epoch=epoch,
                    step=step,
                    loss=total_loss,
                    accuracy_score=total_acc,
                    f1_score=total_f1,
                )
            torch.cuda.empty_cache()

        val_loss = val_acc = val_f1 = 0

        model.eval()
        with torch.no_grad():
            for inputs in tqdm(val_data_loader):
                loss, acc, f1 = evaluation(inputs, model, criterion)
                val_loss += loss.item()
                val_acc += acc
                val_f1 += f1

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./jw-mt5-base.bin")
        print_result(test_type="val", step=len(val_data_loader), loss=val_loss, accuracy_score=val_acc, f1_score=val_f1)
        torch.cuda.empty_cache()


def test(model: PromptForClassification, test_data_loader: PromptDataLoader, criterion: torch.nn.Module) -> None:
    test_loss = test_acc = test_f1 = 0
    with torch.no_grad():
        for inputs in tqdm(test_data_loader):
            loss, acc, f1 = evaluation(inputs, model, criterion)
            test_loss += loss.item()
            test_acc += acc
            test_f1 += f1
    print_result(
        test_type="test", step=len(test_data_loader), loss=test_loss, accuracy_score=test_acc, f1_score=test_f1
    )


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:

    seed_everything(cfg.seed)
    log.info(cfg)
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name=cfg.model.name, model_path=cfg.model.path)
    nli_data_module: PromptNliDataModule = hydra.utils.instantiate(cfg.dataset.nli)

    special_tokens = ["</s>", "<unk>", "<pad>"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    log.info(f"{num_added_tokens}개의 special token 생성")
    log.info(f"new special_token : {special_tokens}")

    template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

    prompt_loader: PromptLoader = hydra.utils.instantiate(cfg.dataset.loader)
    train_data_loader, val_data_loader, test_data_loader = [
        prompt_loader.get_loader(
            dataset=nli_data_module.prompt_input_dataset[data_type],
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
        )
        for data_type in ["train", "val", "test"]
    ]
    verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=2, label_words=[["yes"], ["no"]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"running on : {device}")
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)  # freeze 고려
    prompt_model.to(device)
    prompt_model.load_state_dict(torch.load('./jw-mt5-base.bin'))
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

    test(model=prompt_model, test_data_loader=test_data_loader, criterion=criterion)

    labeled_data_module: PromptLabeledDataModule = hydra.utils.instantiate(cfg.dataset.labeled)
    train_data_loader, val_data_loader, test_data_loader = [
        prompt_loader.get_loader(
            dataset=labeled_data_module.prompt_input_dataset[data_type],
            template=template,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
        )
        for data_type in ["train", "val", "test"]
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

    test(model=prompt_model, test_data_loader=test_data_loader, criterion=criterion)


if __name__ == "__main__":
    main()
