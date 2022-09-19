import time

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.maximize_window()


def translate_with_papago(text: str, source_language: str, target_language: str) -> str:
    """
    trans_lang에 넣는 파라미터 값:
    'en' -> 영어
    'ja&hn=0' -> 일본어
    'zh-CN' -> 중국어(간체)
    """
    try:
        driver.get(f"https://papago.naver.com/?sk={source_language}&tk={target_language}&st={text}")
        time.sleep(3)
        translated_text = driver.find_element(By.XPATH, r'//*[@id="txtTarget"]').text
        return translated_text
    except:  # noqa
        driver.get(f"https://papago.naver.com/?sk={source_language}&tk={target_language}")
        driver.find_element(By.XPATH, r'//*[@id="txtSource"]').send_keys(text)
        time.sleep(3)
        translated_text = driver.find_element(By.XPATH, r'//*[@id="txtTarget"]').text
        return translated_text


def back_translate(text: str) -> str:
    translated_text = translate_with_papago(text, "ko", "en")
    back_translation_text = translate_with_papago(translated_text, "en", "ko")
    return back_translation_text


def back_translation_augmentation(train_csv_path: str) -> None:
    df = pd.read_csv(train_csv_path)
    print(f"previous data size : {len(df)}")
    df["user_answer"] = df["user_answer"].apply(
        lambda x: x.strip().replace("\n", "").replace("\xa0", "").replace("  ", " ")
    )
    df["user_answer"].replace("", np.nan, inplace=True)
    df.dropna(axis=0, subset=["user_answer"], inplace=True)  # 빈 답변 제거
    new_df = pd.read_csv("./augmented_train.csv")

    for idx in new_df.index:
        if new_df.iloc[idx].user_answer == df.iloc[idx].user_answer:
            while True:
                try:
                    print(idx)
                    if len(new_df.iloc[idx].user_answer) < 3:
                        continue
                    back_translated_user_answer = back_translate(new_df.iloc[idx].user_answer)
                    new_df.iloc[idx].user_answer = back_translated_user_answer
                except:  # noqa
                    print(f"에러가 발생했으니 {idx}부터 다시 시작합니다")
                    new_df.to_csv("./augmented_temp.csv", index=False)
                else:
                    break

    new_df = pd.concat([df, new_df])
    print(f"augmented train data size : {len(new_df)}")
    new_df.to_csv("./augmented_train.csv", index=False)
    print("saved at ./augmented_train.csv")


if __name__ == "__main__":
    back_translation_augmentation(train_csv_path="train.csv")
