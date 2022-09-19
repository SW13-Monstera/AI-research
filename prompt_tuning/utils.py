import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import wandb

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


def print_train(loss: float, epoch: int) -> None:
    log.info(f"[Train] Epoch {epoch} loss: {loss} ")
    wandb.log({"train loss": loss})


def print_test(loss: float, accuracy: float, f1_score: float, joint_goal_accuracy: float) -> None:
    log.info(f"[test] loss: {loss} accuracy: {accuracy} f1_score: {f1_score} JGA: {joint_goal_accuracy}")
    wandb.log({"val loss": loss, "accuracy": accuracy, "f1_score": f1_score, "JGA": joint_goal_accuracy})


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


def split_test_dataset(
    dataset_path: str = "../static/labeled_dataset.csv", test_rate: float = 0.2, seed: int = 42
) -> None:
    df = pd.read_csv(dataset_path)
    random.seed(seed)
    answer_id_list = list(df.index)
    random.shuffle(answer_id_list)
    start_val_idx = int(len(answer_id_list) * (1 - test_rate))
    train_id_list, test_id_list = answer_id_list[:start_val_idx], answer_id_list[start_val_idx:]
    train_df, test_df = df.iloc[train_id_list], df.iloc[test_id_list]
    dir_name = os.path.dirname(dataset_path)
    train_df.to_csv(f"{dir_name}/train.csv", index=False)
    test_df.to_csv(f"{dir_name}/test.csv", index=False)


# def translate_with_papago(text: str, source_language: str, target_language: str) -> str:
#     encoding_text = urllib.parse.quote(text)
#     data = f"source={source_language}&target={target_language}&text={encoding_text}"
#     url = "https://openapi.naver.com/v1/papago/n2mt"
#     request = urllib.request.Request(url)
#     request.add_header("X-Naver-Client-Id", NAVER_CLIENT_ID)
#     request.add_header("X-Naver-Client-Secret", NAVER_CLIENT_SECRET)
#     response = urllib.request.urlopen(request, data=data.encode("utf-8"))
#     rescode = response.getcode()
#     if rescode == 200:
#         response_body = eval(response.read().decode("utf-8").replace("null", "None"))
#         return response_body["message"]["result"]["translatedText"]
#     else:
#         log.info("Error Code:" + rescode)
#         return ""
#
#
# def back_translate(text: str) -> str:
#     translated_text = translate_with_papago(text, "ko", "en")
#     time.sleep(1)
#     back_translation_text = translate_with_papago(translated_text, "en", "ko")
#     time.sleep(1)
#     return back_translation_text
#
#
# def back_translation_augmentation(train_csv_path: str) -> None:
#     df = pd.read_csv(train_csv_path)
#     log.info(f"previous data size : {len(df)}")
#     df["user_answer"] = df["user_answer"].apply(
#         lambda x: x.strip().replace("\n", "").replace("\xa0", "").replace("  ", " ")
#     )
#     df["user_answer"].replace("", np.nan, inplace=True)
#     df.dropna(axis=0, subset=["user_answer"], inplace=True)  # 빈 답변 제거
#     new_df = df.copy()
#     for idx in new_df.index:
#         back_translated_user_answer = back_translate(new_df.iloc[idx].user_answer)
#         new_df.iloc[idx].user_answer = back_translated_user_answer
#     new_df = pd.concat([df, new_df])
#     dir_name = os.path.dirname(train_csv_path)
#     log.info(f"augmented train data size : {len(new_df)}")
#     new_df.to_csv(f"{dir_name}/augmented_train.csv", index=False)
