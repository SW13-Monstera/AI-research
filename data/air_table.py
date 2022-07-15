from typing import Tuple

from airtable import Airtable
from collections import defaultdict
from core import config as settings


def check_valid_score(data: dict) -> None:
    for key in data:
        assert data[key] == 5, f"problem id : {key}\n 키워드, 유사도 점수의 합은 항상 각각 5점 이어야 합니다!!"


def get_keyword() -> Tuple[dict, defaultdict]:
    keyword_view = Airtable(settings.AIR_TABLE_APP_NAME, "keyword", settings.AIR_TABLE_API_KEY).get_all()
    problem_keywords = defaultdict(list)
    problem_keywords_score = defaultdict(int)
    keywords = {}
    for keyword in keyword_view:
        if keyword['fields'] == {}:
            continue
        problem_id = keyword['fields']['problem_id'][0]
        score = keyword['fields']['score']
        keyword_id = keyword['fields']['keyword_id']
        problem_keywords_score[problem_id] += score
        problem_keywords[problem_id].append(keyword_id)
        keywords[keyword_id] = keyword['fields']['keyword_content']

    check_valid_score(problem_keywords_score)
    return keywords, problem_keywords


def get_similarity() -> Tuple[dict, defaultdict]:
    similarity_view = Airtable(settings.AIR_TABLE_APP_NAME, "similarity", settings.AIR_TABLE_API_KEY).get_all()
    problem_similarities = defaultdict(list)
    problem_similarities_score = defaultdict(int)
    similarities = {}
    for similarity in similarity_view:
        if similarity['fields'] == {}:
            continue
        problem_id = similarity['fields']['problem_id'][0]
        score = similarity['fields']['score']
        similarity_id = similarity['fields']['sim_id']
        problem_similarities_score[problem_id] += score
        problem_similarities[problem_id].append(similarity_id)
        similarities[similarity_id] = similarity['fields']['sim_content']

    check_valid_score(problem_similarities_score)
    return similarities, problem_similarities


def get_required_labelling_record() -> list:
    table = Airtable(settings.AIR_TABLE_APP_NAME, "main", settings.AIR_TABLE_API_KEY)
    records = table.get_all(view="채점이 필요한 데이터")
    keyword_dict, problem_keywords = get_keyword()
    similarity_dict, problem_similarities = get_similarity()
    convert_data = []
    for record in records:
        json_data = record['fields']
        problem_id = record['fields']['problem_id'][0]

        keyword_ids = problem_keywords[problem_id]
        keyword_list = [{"alias": keyword_id, "value": keyword_dict[keyword_id]} for keyword_id in keyword_ids]
        similarity_ids = problem_similarities[problem_id]
        similarity_list = []
        for similarity_id in similarity_ids:
            similarity_list.append({"alias": similarity_id, "value": similarity_dict[similarity_id]})
        json_data['keyword_list'] = keyword_list
        json_data['sim_list'] = similarity_list
        json_data['problem_id'] = json_data['problem_id'][0]
        json_data['assign'] = json_data['assign'][0]
        json_data['problem'] = json_data['problem'][0]
        json_data.pop('sim_content')
        json_data.pop('keyword_content')
        convert_data.append(json_data)
    return convert_data
