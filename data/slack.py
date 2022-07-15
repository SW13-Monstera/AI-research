from slack_sdk import WebClient
from slack_sdk.web import SlackResponse


class SlackAPI:
    """
    슬랙 API 핸들러
    """

    def __init__(self, token) -> None:
        # 슬랙 클라이언트 인스턴스 생성
        self.client = WebClient(token)

    def get_channel_id(self, channel_name) -> str:
        """
        슬랙 채널ID 조회
        """
        # conversations_list() 메서드 호출
        result = self.client.conversations_list()
        # 채널 정보 딕셔너리 리스트
        channels = result.data['channels']
        # 채널 명이 'test'인 채널 딕셔너리 쿼리
        channel = list(filter(lambda c: c["name"] == channel_name, channels))[0]
        # 채널ID 파싱
        channel_id = channel["id"]
        return channel_id

    def get_message_ts(self, channel_id, query) -> list:
        """
        슬랙 채널 내 메세지 조회
        """
        # conversations_history() 메서드 호출
        result = self.client.conversations_history(channel=channel_id)
        # 채널 내 메세지 정보 딕셔너리 리스트
        messages = result.data['messages']
        # 채널 내 메세지가 query와 일치하는 메세지 딕셔너리 쿼리
        message = list(filter(lambda m: m["text"] == query, messages))[0]
        # 해당 메세지ts 파싱
        message_ts = message["ts"]
        return message_ts

    def post_thread_message(self, channel_id, message_ts, text) -> SlackResponse:
        """
        슬랙 채널 내 메세지의 Thread에 댓글 달기
        """
        # chat_postMessage() 메서드 호출
        result = self.client.chat_postMessage(
            channel=channel_id,
            text=text,
            thread_ts=message_ts
        )
        return result

    def post_message(self, channel_id, text) -> SlackResponse:
        result = self.client.chat_postMessage(
            channel=channel_id,
            text=text
        )
        return result

    def get_username_messages(self, channel_id, username) -> list:
        result = self.client.conversations_history(channel=channel_id)
        messages = result.data['messages']

        user_messages = []
        for message in messages:
            if 'username' in message and username == message['username']:
                print(message['username'])
                print(message['ts'])
                print(message['text'][:20])
                user_messages.append(message)
        user_messages.sort(key=lambda msg: float(msg['ts']))

        return user_messages


def air_table_text_parsing(text: str) -> list:

    def remove_newline(_text: str) -> str:
        start, end = 0, len(_text)
        for i in range(len(_text)):
            if _text[i] != '\n':
                start = i
                break

        for i in range(len(_text) - 1, -1, -1):
            if _text[i] != '\n':
                end = i
                break
        return _text[start:end+1]

    answer_data_list = []
    split_by_data = text.split('*data_id:*')[1:]
    for data in split_by_data:
        data_id, data = data.split('*problem_id:*')
        problem_id, data = data.split('*assign:*')
        assign, data = data.split('*keyword_content:*')
        keyword_content, data = data.split('*sim_content:*')
        sim_content, data = data.split('*problem:*')
        problem, data = data.split('*user_answer:*')
        user_answer, data = data.split('*reference:*')
        reference, data = data.split('*is_uploaded:*')
        answer_data = {
            "data_id": remove_newline(data_id),
            "problem_id": remove_newline(problem_id),
            "assign": remove_newline(assign),
            "keyword_content": remove_newline(keyword_content),
            "sim_content": remove_newline(sim_content),
            'problem': remove_newline(problem),
            'user_answer': remove_newline(user_answer),
            'reference': remove_newline(reference)
        }
        answer_data_list.append(answer_data)
    return answer_data_list

