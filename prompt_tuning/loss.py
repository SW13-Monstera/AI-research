import logging
from typing import Tuple

import numpy as np
import torch
from openprompt import PromptForClassification
from openprompt.data_utils import InputFeatures
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor

log = logging.getLogger("__main__")


def calculate_metric(labels, predicts) -> Tuple[float, float, float]:
    acc = accuracy_score(labels, predicts)
    f1 = f1_score(labels, predicts)
    auc = roc_auc_score(labels, predicts)
    return acc, f1, auc


def get_predicts(inputs: InputFeatures, prompt_model: PromptForClassification) -> Tensor:
    device = prompt_model.device
    inputs = inputs.to(device)
    logits = prompt_model(inputs)
    predicts = torch.argmax(logits, dim=1).cpu().numpy()
    return predicts


class EarlyStopping:
    # 주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # validation loss가 감소하면 모델을 저장
        if self.verbose:
            log.info(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
