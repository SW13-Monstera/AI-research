from openprompt import PromptForClassification
from openprompt.data_utils import InputFeatures
from torch import Tensor
from torch.nn.modules import Module


def get_loss(inputs: InputFeatures, prompt_model: PromptForClassification, criterion: Module) -> Tensor:
    device = prompt_model.device
    inputs = inputs.to(device)
    logits = prompt_model(inputs)
    labels = inputs.label
    loss = criterion(logits, labels)
    return loss
