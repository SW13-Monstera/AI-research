import logging

import boto3
from botocore.exceptions import ClientError

session = boto3.Session()
secret_manager = session.client(service_name="secretsmanager", region_name="ap-northeast-2")
log = logging.getLogger("__main__")
log.setLevel(logging.INFO)


def get_secret() -> dict:
    secret_name = "CS-Broker"

    try:
        get_secret_value_response = secret_manager.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        log.error(e)
        if e.response["Error"]["Code"] == "DecryptionFailureException":
            raise e
        elif e.response["Error"]["Code"] == "InternalServiceErrorException":
            raise e
        elif e.response["Error"]["Code"] == "InvalidParameterException":
            raise e
        elif e.response["Error"]["Code"] == "InvalidRequestException":
            raise e
        elif e.response["Error"]["Code"] == "ResourceNotFoundException":
            raise e
    else:
        secret = get_secret_value_response["SecretString"]
        return eval(secret)


secret = get_secret()
SLACK_COLLECTOR_BOT_TOKEN = secret["SLACK_COLLECTOR_BOT_TOKEN"]
AIR_TABLE_APP_NAME = secret["AIR_TABLE_APP_NAME"]
AIR_TABLE_API_KEY = secret["AIR_TABLE_API_KEY"]
LABEL_STUDIO_ACCESS_TOKEN = secret["LABEL_STUDIO_ACCESS_TOKEN"]
LABEL_STUDIO_URL = secret["LABEL_STUDIO_URL"]
HUGGING_FACE_ACCESS_TOKEN = secret["HUGGING_FACE_ACCESS_TOKEN"]
