import random

import torch
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.data_utils import InputExample
from openprompt.plms import T5TokenizerWrapper, load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer

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
    plm, tokenizer, model_config, _ = load_plm(
        model_name="t5",
        model_path=pretrain_model_path,
    )
    special_tokens = ["</s>", "<unk>", "<pad>"]
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"{num_added_tokens}개의 special token 생성")
    print(special_tokens)

    template_text = '{"placeholder":"text_a"} Question: {"placeholder":"text_b"}? Is it correct? {"mask"}.'
    template = ManualTemplate(tokenizer=tokenizer, text=template_text)

    wrapped_example = template.wrap_one_example(prompt_input_dataset["train"][0])
    print(wrapped_example)

    wrapped_t5_tokenizer = T5TokenizerWrapper(
        max_seq_length=128,
        decoder_max_length=3,
        tokenizer=tokenizer,
        truncate_method="head",
    )

    tokenized_example = wrapped_t5_tokenizer.tokenize_one_example(
        wrapped_example, teacher_forcing=False
    )

    model_inputs = {}
    for split in ["train", "validation", "test"]:
        model_inputs[split] = []
        for data in prompt_input_dataset[split]:
            tokenized_example = wrapped_t5_tokenizer.tokenize_one_example(
                wrapped_example=template.wrap_one_example(data), teacher_forcing=False
            )
            model_inputs[split].append(tokenized_example)

    train_data_loader = PromptDataLoader(
        dataset=prompt_input_dataset["train"],
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=T5TokenizerWrapper,
        max_seq_length=256,  # openprompt 에서 time hinting 실수한 듯 -> open source 기여 각?
        decoder_max_length=3,
        batch_size=4,
        # shuffle=True,  # seed 고정 되는지 확인 후 True 설정
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="head",
    )

    verbalizer = ManualVerbalizer(
        tokenizer=tokenizer,
        num_classes=3,  # 여기도 string으로 되어있는데 내부 고민
        label_words=[["yes"], ["no"], ["maybe"]],
    )
    print(verbalizer.label_words)
    logits = torch.randn(2, len(tokenizer))
    print(verbalizer.process_logits(logits))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_model = PromptForClassification(
        plm=plm, template=template, verbalizer=verbalizer
    )  # freeze 고려
    prompt_model.to(device)

    loss = torch.nn.CrossEntropyLoss()  # loss 생각해보기
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in prompt_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
