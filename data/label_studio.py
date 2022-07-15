from label_studio_sdk import Client


class LabelStudioAPI:
    """
    label studio API 핸들러
    """
    def __init__(self, url: str, api_key: str, project_id: int = 43) -> None:
        self.client = Client(url=url, api_key=api_key)
        self.project = self.client.get_project(project_id)

    def import_tasks(self, dataset: list) -> None:
        exist_data_id_set = self._get_data_id_set()
        new_data = []
        for data in dataset:
            if data['data_id'] not in exist_data_id_set:
                new_data.append(data)

        if new_data:  # Todo: 로깅 추가
            print(self.project.import_tasks(new_data))

    def _get_data_id_set(self) -> set:
        data_id_set = set(task['data']['data_id'] for task in self.project.get_tasks())
        return data_id_set
