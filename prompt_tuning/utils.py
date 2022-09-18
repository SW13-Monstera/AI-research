import logging
import random
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("__main__")
log.setLevel(logging.INFO)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def print_result(
    test_type: str, step: int, loss: float, accuracy_score: float, f1_score: float, epoch: Optional[int] = None
) -> None:
    if step != 0:
        log.info(
            (
                f"[{test_type}] "
                f"Epoch {epoch if epoch is not None else ''} "
                f"{test_type} loss: {loss / step} "
                f"accuracy: {accuracy_score / step} "
                f"f1_score: {f1_score / step}"
            )
        )
