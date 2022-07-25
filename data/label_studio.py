from label_studio_sdk import Client
import requests

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
            if data['data_id'] not in exist_data_id_set:
                new_data.append(data)

        if new_data:  # Todo: 로깅 추가
            import_tasks = self.project.import_tasks(new_data)
            print(f"{len(import_tasks)}개의 태스크가 생성되었습니다.")
        else:
            print(f"0개의 테스크가 생성되었습니다.")

    def _get_data_id_set(self) -> set:
        data_id_set = set(task['data']['data_id'] for task in self.project.get_tasks())
        return data_id_set

    def get_task_ids(self, problem: str = "[운영체제 2]") -> list:
        task_ids = [task['id'] for task in self.project.get_tasks() if problem in task['data']['problem']]
        return task_ids

    def delete_tasks(self, id_list: list) -> None:
        for _id in id_list:
            requests.delete(
                f"{self.client.url}/api/tasks/{_id}/",
                headers={"Authorization": f"Token {self.client.api_key}"}
            )
        print(f"{len(id_list)}개의 task 삭제 완료!")
