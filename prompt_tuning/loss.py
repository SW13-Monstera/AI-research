import logging
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

log = logging.getLogger("__main__")


def calculate_metric(labels, predicts) -> Tuple[float, float]:
    acc = accuracy_score(labels, predicts)
    f1 = f1_score(labels, predicts)
    return acc, f1


class EarlyStopping:
    LOSS = "loss"
    JGA = "joint_goal_accuracy"
    ACC = "accuracy"

    def __init__(
        self,
        standard: Optional[str],
        patience: int = 7,
        verbose: bool = False,
        delta: int = 0,
        path: str = "checkpoint.pt",
    ):
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
        self.best_acc = 0
        self.best_joint_goal_accuracy = 0
        self.standard = standard if standard else self.JGA

    def __call__(self, model, standard_value):

        if self.standard is self.JGA:
            if self.best_joint_goal_accuracy > standard_value:
                self.counter += 1
                log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if self.verbose:
                    log.info(
                        f"New best model JGA : ({self.best_joint_goal_accuracy:.6f}"
                        f" --> {standard_value:.6f}) Saving model"
                    )
                torch.save(model.state_dict(), self.path)
                self.best_joint_goal_accuracy = standard_value
                self.counter = 0

        elif self.standard is self.LOSS:
            if self.val_loss_min < standard_value:
                self.counter += 1
                log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:

                if self.verbose:
                    log.info(f"New best model loss : ({self.val_loss_min:.6f} --> {standard_value:.6f}) Saving model")
                torch.save(model.state_dict(), self.path)
                self.val_loss_min = standard_value
                self.counter = 0
