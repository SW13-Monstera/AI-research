from pydantic import BaseModel


class UserAnswer(BaseModel):
    problem_id: str
    problem: str
    assign: str
    user_answer: str
    scoring_criterion: list
    correct_scoring_criterion: list
    keyword_criterion: list
    correct_keyword_criterion: list
    annotator: str
