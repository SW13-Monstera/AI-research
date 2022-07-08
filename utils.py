import pandas as pd
import argparse
import json


def preprocessing_csv(answer_path: str, keyword_path: str, similarity_path: str):
    """
    {
      "data": {
        "variants": [
          { "value": "Do or doughnut. There is no try." },
          { "value": "Do or do not. There is no trial." },
          { "value": "Do or do not. There is no try." },
          { "value": "Duo do not. There is no try." }
        ]
      }
    }
    """
    convert_data = []
    answer_pd = pd.read_csv(answer_path)
    answer_pd.drop(columns=['keyword_content', 'sim_content'], inplace=True)
    keyword_pd = pd.read_csv(keyword_path)
    similarity_pd = pd.read_csv(similarity_path)

    for i, answer_row in answer_pd.iterrows():
        json_data = dict(answer_row)
        keyword_list = []
        keyword_score_list = []
        sim_list = []
        sim_score_list = []
        for i, keyword_row in keyword_pd.iterrows():
            if keyword_row['problem_id'] == answer_row['problem_id']:
                keyword_list.append(keyword_row['keyword_content'])
                keyword_score_list.append(keyword_row['score'])

        for i, sim_row in similarity_pd.iterrows():
            if sim_row['problem_id'] == answer_row['problem_id']:
                sim_list.append(sim_row['sim_content'])
                sim_score_list.append(sim_row['score'])
        json_data['keyword_list'] = keyword_list
        json_data['keyword_score_list'] = keyword_score_list
        json_data['sim_list'] = sim_list
        json_data['sim_score_list'] = sim_score_list
        convert_data.append(json_data)

    with open('answer.json', 'w') as f:
        json.dump(convert_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, default='./answer.csv')
    parser.add_argument('--keyword_path', type=str, default='./keyword.csv')
    parser.add_argument('--similarity_path', type=str, default='./similarity.csv')
    args = parser.parse_args()
    preprocessing_csv(args.answer_path, args.keyword_path, args.similarity_path)