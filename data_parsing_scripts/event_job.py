import csv
from datetime import datetime

from core import config as settings
from data_parsing_scripts.air_table import get_required_labelling_record
from data_parsing_scripts.dataset import UserAnswer
from data_parsing_scripts.label_studio import LabelStudioAPI


def upload_unlabeled_data_to_label_studio_from_air_table() -> None:
    label_studio = LabelStudioAPI(settings.LABEL_STUDIO_URL, settings.LABEL_STUDIO_ACCESS_TOKEN)
    unlabeled_dataset = get_required_labelling_record()
    label_studio.import_tasks(unlabeled_dataset)


def delete_problem_in_label_studio(problem: str) -> None:
    label_studio = LabelStudioAPI(settings.LABEL_STUDIO_URL, settings.LABEL_STUDIO_ACCESS_TOKEN)
    task_ids = label_studio.get_task_ids(problem)
    stop_flag = input(f"삭제 하려는 문제가 {problem}이 맞나요? 이 실행은 되돌릴 수 없습니다!! 만약 취소하려면 0을 눌러주시고 실행하려면 0이아닌 입력을 해주세요")
    if stop_flag:
        label_studio.delete_tasks(task_ids)


def transform_required_csv_form_from_label_studio() -> None:
    label_studio = LabelStudioAPI(settings.LABEL_STUDIO_URL, settings.LABEL_STUDIO_ACCESS_TOKEN)
    labeled_tasks = label_studio.get_labeled_tasks()
    field_list = list(UserAnswer.__fields__.keys())
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    with open(f"{now}_user_answer.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(field_list)
        for labeled_task in labeled_tasks:
            writer.writerow([getattr(labeled_task, field) for field in field_list])


if __name__ == "__main__":
    transform_required_csv_form_from_label_studio()
