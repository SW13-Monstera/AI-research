from collections import defaultdict
from typing import List

from numpy import ndarray
from sklearn.metrics import accuracy_score, f1_score


class Evaluator:
    def __init__(self):
        self.prediction_dict = defaultdict(list)
        self.label_dict = defaultdict(list)
        self._init()

    def _init(self):
        self.joint_goal_hit = 0
        self.all_hit = 0
        self.f1_sum = 0
        self.acc_sum = 0
        self.loss_sum = 0

    def save(self, labels: ndarray, predicts: ndarray, guids: ndarray, loss: float) -> None:
        for label, predict, guid in zip(labels, predicts, guids):
            self.prediction_dict[guid].append(predict)
            self.label_dict[guid].append(label)
        self.loss_sum += loss

    def compute(self) -> None:
        self._init()
        for guid in self.prediction_dict:
            labels = self.label_dict[guid]
            predicts = self.prediction_dict[guid]
            self._update(labels, predicts)

    def _update(self, labels: List[int], predicts: List[int]) -> None:
        self.all_hit += 1
        if labels == predicts:
            self.joint_goal_hit += 1
        self.acc_sum += accuracy_score(labels, predicts)
        self.f1_sum += f1_score(labels, predicts)

    @property
    def f1_score(self) -> float:
        return self.f1_sum / self.all_hit if self.all_hit != 0 else 0

    @property
    def acc(self) -> float:
        return self.acc_sum / self.all_hit if self.all_hit != 0 else 0

    @property
    def joint_goal_acc(self) -> float:
        return self.joint_goal_hit / self.all_hit if self.all_hit != 0 else 0

    @property
    def loss(self) -> float:
        return self.loss_sum / self.all_hit if self.all_hit != 0 else 0
