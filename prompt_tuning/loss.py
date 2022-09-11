from typing import Tuple

import torch
from openprompt import PromptForClassification
from openprompt.data_utils import InputFeatures
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.nn.modules import Module


def evaluation(
    inputs: InputFeatures, prompt_model: PromptForClassification, criterion: Module
) -> Tuple[Tensor, float, float]:
    device = prompt_model.device
    inputs = inputs.to(device)
    logits = prompt_model(inputs)
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = inputs.label.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    labels = inputs.label
    loss = criterion(logits, labels)
    return loss, acc, f1
