import logging
import os
import random
from typing import Optional

import numpy as np
import torch
from openprompt import PromptForClassification

from core.config import session

log = logging.getLogger("__main__")
log.setLevel(logging.INFO)
s3 = session.client(service_name="s3", region_name="ap-northeast-2")


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


def upload_model_to_s3(
    model: PromptForClassification,
    bucket: str = "cs-broker-bucket",
    folder: str = "ai-models/default",
    model_name: str = "model.pth",
):
    exist_bucket_set = set((bucket.get("Name") for bucket in s3.list_buckets().get("Buckets", [])))
    if bucket not in exist_bucket_set:
        s3.create_bucket(Bucket=bucket)
        log.info(f"{bucket} bucket 생성")

    local_path = f"./{model_name}"
    log.info(f"{local_path}에 임시 모델이 생성")
    torch.save(model.state_dict(), local_path)

    log.info(f"{os.path.join(bucket, folder, model_name)}에 모델 업로드중")
    s3.upload_file(local_path, bucket, os.path.join(folder, model_name))
    log.info("모델 업로드 완료")

    log.info(f"{local_path}의 임시 모델 삭제")
    os.remove(local_path)
