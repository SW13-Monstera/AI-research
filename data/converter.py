import pandas as pd
import argparse
import json


def preprocessing_csv(answer_path: str, keyword_path: str, similarity_path: str):
    """
    {
      "data": {
        "keyword_list": [
          { "alias": "a", "value": "Do or doughnut. There is no try." },
          { "alias": "b", "value": "Do or do not. There is no trial." },
          { "alias": "c", "value": "Do or do not. There is no try." },
          { "alias": "d", "value": "Duo do not. There is no try." }
        ]
      }
    }
    """
    convert_data = []
    answer_pd = pd.read_csv(answer_path)
    answer_pd.drop(columns=['keyword_content', 'sim_content'], inplace=True)
    keyword_pd = pd.read_csv(keyword_path)
    similarity_pd = pd.read_csv(similarity_path)

    for _, answer_row in answer_pd.iterrows():
        json_data = dict(answer_row.where(pd.notnull(answer_row), None))
        keyword_list = []
        sim_list = []
        score = []
        for i, keyword_row in keyword_pd.iterrows():
            if keyword_row['problem_id'] == answer_row['problem_id']:
                keyword_list.append({"alias": keyword_row['keyword_id'], "value": keyword_row['keyword_content']})
                score.append(keyword_row['score'])
        assert sum(score) == 5, "keyword score의 합은 5점 이어야 합니다!!"
        score.clear()
        for i, sim_row in similarity_pd.iterrows():
            if sim_row['problem_id'] == answer_row['problem_id']:
                sim_list.append({"alias": sim_row['sim_id'], "value": sim_row['sim_content']})
                score.append(sim_row['score'])
        assert sum(score) == 5, f"{answer_row}similarity score의 합은 5점 이어야 합니다!!"
        json_data['keyword_list'] = keyword_list
        json_data['sim_list'] = sim_list
        convert_data.append(json_data)

    with open('static/answer.json', 'w') as f:
        json.dump(convert_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, default='static/answer.csv')
    parser.add_argument('--keyword_path', type=str, default='static/keyword.csv')
    parser.add_argument('--similarity_path', type=str, default='static/similarity.csv')
    args = parser.parse_args()
    preprocessing_csv(args.answer_path, args.keyword_path, args.similarity_path)
