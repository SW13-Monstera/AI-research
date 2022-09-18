import logging
import random
import urllib
from typing import Optional

import numpy as np
import torch
import wandb

from core.config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, session

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
    test_type: str,
    step: int,
    loss: float,
    accuracy_score: Optional[float] = None,
    f1_score: Optional[float] = None,
    epoch: Optional[int] = None,
) -> None:
    if step != 0:
        if test_type == "train":
            log_string = f"[{test_type}] Epoch {epoch} loss: {loss / step} "
            log.info(log_string)
            wandb.log({"train loss": loss / step})
        else:
            log_string = (
                f"[{test_type}] Epoch {epoch} loss: {loss / step}"
                f" accuracy: {accuracy_score / step} f1_score: {f1_score / step}"
            )
            log.info(log_string)
            wandb.log(
                {
                    "val loss": loss / step,
                    "accuracy": accuracy_score / step,
                    "f1_score": f1_score / step,
                }
            )


def upload_model_to_s3(
    local_path: str = "checkpoint.pt",
    bucket: str = "cs-broker-bucket",
    folder: str = "ai-models/default",
    model_name: str = "model.pth",
):
    exist_bucket_set = set((bucket.get("Name") for bucket in s3.list_buckets().get("Buckets", [])))
    if bucket not in exist_bucket_set:
        s3.create_bucket(Bucket=bucket)
        log.info(f"{bucket} bucket 생성")

    log.info(f"{bucket}/{folder}/{model_name}에 모델 업로드중")
    s3.upload_file(local_path, bucket, f"{folder}/{model_name}")
    log.info("모델 업로드 완료")


def translate_with_papago(text: str, source_language: str, target_language: str) -> str:
    encoding_text = urllib.parse.quote(text)
    data = f"source={source_language}&target={target_language}&text={encoding_text}"
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if rescode == 200:
        response_body = eval(response.read().decode("utf-8").replace("null", "None"))
        return response_body["message"]["result"]["translatedText"]
    else:
        log.info("Error Code:" + rescode)
        return ""


def back_translate(text: str) -> str:
    translated_text = translate_with_papago(text, "ko", "en")
    back_translation_text = translate_with_papago(translated_text, "en", "ko")
    return back_translation_text
