import pyrootutils
import torch
from loss import get_loss
from openprompt import PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from torch.optim import AdamW
from utils import seed_everything

from prompt_tuning.dataset import PromptDataModule

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)


if __name__ == "__main__":
    #######################
    # hydra로 관리 할 hyper parameter들
    # data
    seed = 42
    task = "nli"
    dataset_path = "klue"
    dataset_name = "nli"
    split_rate = (8, 1, 1)
    seed_everything(seed)
    max_seq_length = 256
    decoder_max_length = 3
    batch_size = 4
    teacher_forcing = False
    predict_eos_token = False
    truncate_method = "head"

    # model
    model_name = "t5"
    pretrain_model_path = "google/mt5-base"

    # train
    epochs = 10
    lr = 1e-4
    logging_steps = 100
    weight_decay = 0.1

    #########################
    data_module = PromptDataModule(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        seed=seed,
        split_rate=split_rate,
        max_seq_length=max_seq_length,
        decoder_max_length=decoder_max_length,
        batch_size=batch_size,
        teacher_forcing=teacher_forcing,
        predict_eos_token=predict_eos_token,
        truncate_method=truncate_method,
    )
    plm, tokenizer, model_config, WrapperClass = load_plm(
        model_name=model_name,
        model_path=pretrain_model_path,
    )
    special_tokens = ["</s>", "<unk>", "<pad>"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"{num_added_tokens}개의 special token 생성")
    print(special_tokens)

    template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

    train_data_loader, val_data_loader, test_data_loader = data_module.get_data_loader(
        template=template, tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass
    )

    verbalizer = ManualVerbalizer(tokenizer=tokenizer, num_classes=3, label_words=[["yes"], ["no"], ["maybe"]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_model = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer)  # freeze 고려
    prompt_model.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # loss 생각해보기
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        prompt_model.train()
        for step, inputs in enumerate(train_data_loader):
            loss = get_loss(inputs, prompt_model, criterion)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step % logging_steps == 0:
                print(f"Epoch {epoch}, step: {step} average loss: {total_loss / (step + 1)}")

        print("validation check start")
        validation_loss = 0
        prompt_model.eval()
        for step, inputs in enumerate(val_data_loader):
            loss = get_loss(inputs, prompt_model, criterion)
            validation_loss += loss.item()

        print(f"Epoch {epoch}, validation loss: {validation_loss / len(val_data_loader)}")

    # final testing
    test_loss = 0
    for step, inputs in enumerate(test_data_loader):  # Todo: inputs부터 loss까지 모듈화하기
        loss = get_loss(inputs, prompt_model, criterion)
        test_loss += loss.item()

    print(f"Final test loss: {test_loss / len(test_data_loader)}")
