from data.air_table import get_required_labelling_record
from data.label_studio import LabelStudioAPI
from core import config as settings


def upload_unlabeled_data_to_label_studio_from_air_table() -> None:
    label_studio = LabelStudioAPI(settings.LABEL_STUDIO_URL, settings.LABEL_STUDIO_ACCESS_TOKEN)
    unlabeled_dataset = get_required_labelling_record()
    label_studio.import_tasks(unlabeled_dataset)

if __name__ == '__main__':
    upload_unlabeled_data_to_label_studio_from_air_table()