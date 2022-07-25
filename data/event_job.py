from data.air_table import get_required_labelling_record
from data.label_studio import LabelStudioAPI
from core import config as settings


def upload_unlabeled_data_to_label_studio_from_air_table() -> None:
    label_studio = LabelStudioAPI(settings.LABEL_STUDIO_URL, settings.LABEL_STUDIO_ACCESS_TOKEN)
    unlabeled_dataset = get_required_labelling_record()
    label_studio.import_tasks(unlabeled_dataset)


def delete_problem_in_label_studio(problem: str) -> None:
    label_studio = LabelStudioAPI(settings.LABEL_STUDIO_URL, settings.LABEL_STUDIO_ACCESS_TOKEN)
    task_ids = label_studio.get_task_ids(problem)
    stop_flag = input(
        f"삭제 하려는 문제가 {problem}이 맞나요? 이 실행은 되돌릴 수 없습니다!! 만약 취소하려면 0을 눌러주시고 실행하려면 0이아닌 입력을 해주세요"
    )
    if stop_flag:
        label_studio.delete_tasks(task_ids)
