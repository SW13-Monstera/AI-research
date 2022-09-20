import logging
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

log = logging.getLogger("__main__")


def calculate_metric(labels, predicts) -> Tuple[float, float]:
    acc = accuracy_score(labels, predicts)
    f1 = f1_score(labels, predicts)
    return acc, f1


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
        self.best_acc = 0
        self.best_joint_goal_accuracy = 0

    def __call__(self, model, joint_goal_accuracy):

        if self.best_joint_goal_accuracy > joint_goal_accuracy:
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(joint_goal_accuracy, model)
            self.counter = 0
        # if self.best_score is None:
        #     self.best_score = score
        # elif score < self.best_score + self.delta:
        #     self.counter += 1
        #     log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        # else:
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model)
        #     self.counter = 0
        # self.best_acc = max(self.best_score, acc)

    def save_checkpoint(self, joint_goal_accuracy, model):
        # validation loss가 감소하면 모델을 저장
        if self.verbose:
            log.info(
                f"New best model JGA : ({self.best_joint_goal_accuracy:.6f} --> {joint_goal_accuracy:.6f}) Saving model"
            )
        torch.save(model.state_dict(), self.path)
        self.best_joint_goal_accuracy = joint_goal_accuracy
