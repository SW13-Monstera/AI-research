from adamp import AdamP
from torch.optim import Adam, AdamW


def get_optimizer(optimizer, optimizer_grouped_parameters, learning_rate):
    if optimizer == "adamp":
        return AdamP(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer == "adam":
        return Adam(optimizer_grouped_parameters, lr=learning_rate)
    elif optimizer == "adamw":
        return AdamW(optimizer_grouped_parameters, lr=learning_rate)
