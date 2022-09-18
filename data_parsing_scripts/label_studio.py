import requests
from label_studio_sdk import Client

from data_parsing_scripts.dataset import UserAnswer

SCORING_ANNOTATION = "dynamic_choices2"
KEYWORD_ANNOTATION = "dynamic_choices"


class LabelStudioAPI:
    """
    label studio API 핸들러
    """

    def __init__(self, url: str, api_key: str, project_id: int = 49) -> None:
        self.client = Client(url=url, api_key=api_key)
        self.project = self.client.get_project(project_id)

    def import_tasks(self, dataset: list) -> None:
        exist_data_id_set = self._get_data_id_set()
        new_data = []
        for data in dataset:
            if data["data_id"] not in exist_data_id_set:
                new_data.append(data)

        if new_data:  # Todo: 로깅 추가
            import_tasks = self.project.import_tasks(new_data)
            print(f"{len(import_tasks)}개의 태스크가 생성되었습니다.")
        else:
            print("0개의 테스크가 생성되었습니다.")

    def _get_data_id_set(self) -> set:
        data_id_set = set(task["data"]["data_id"] for task in self.project.get_tasks())
        return data_id_set

    def get_task_ids(self, problem: str = "[운영체제 2]") -> list:
        task_ids = [task["id"] for task in self.project.get_tasks() if problem in task["data"]["problem"]]
        return task_ids

    def delete_tasks(self, id_list: list) -> None:
        for _id in id_list:
            requests.delete(
                f"{self.client.url}/api/tasks/{_id}/",
                headers={"Authorization": f"Token {self.client.api_key}"},
            )
        print(f"{len(id_list)}개의 task 삭제 완료!")

    def get_labeled_tasks(self) -> list:
        labeled_tasks = []
        for task in self.project.get_labeled_tasks():
            scoring_criterion_dict = {criterion["alias"]: criterion["value"] for criterion in task["data"]["sim_list"]}
            keyword_criterion_dict = {
                criterion["alias"]: criterion["value"] for criterion in task["data"]["keyword_list"]
            }
            annotations: dict = task["annotations"][-1]["result"]

            # 선택된 후보가 여러개인 경우, 한개인 경우, 없는 경우 세가지로 나뉨
            scoring_annotation, keyword_annotation = [], []

            for annotation in annotations:
                if isinstance(annotation["value"], dict):  # 선택지가 여러개인 경우
                    for choice, *_ in annotation["value"]["choices"]:
                        if annotation["from_name"] == SCORING_ANNOTATION:
                            scoring_annotation.append(scoring_criterion_dict[choice])
                        elif annotation["from_name"] == KEYWORD_ANNOTATION:
                            keyword_annotation.append(keyword_criterion_dict[choice])
                elif isinstance(annotation["value"], list):  # 한개인 경우
                    if annotation["from_name"] == SCORING_ANNOTATION:
                        scoring_annotation.append(scoring_criterion_dict[annotation["value"][0]])
                    elif annotation["from_name"] == KEYWORD_ANNOTATION:
                        keyword_annotation.append(keyword_criterion_dict[annotation["value"][0]])
            user_answer = UserAnswer(
                problem_id=task["data"]["problem_id"],
                problem=task["data"]["problem"],
                assign=task["data"]["assign"],
                user_answer=task["data"]["user_answer"],
                scoring_criterion=list(scoring_criterion_dict.values()),
                correct_scoring_criterion=scoring_annotation,
                keyword_criterion=list(keyword_criterion_dict.values()),
                correct_keyword_criterion=keyword_annotation,
                annotator=task["annotations"][-1]["created_username"].split(",")[0].strip(),
            )
            labeled_tasks.append(user_answer)
        return labeled_tasks
