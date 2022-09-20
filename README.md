# AI-research
AI research repository

상태관리 : hydra

pip3 install openprompt
pip3 install torch
pip3 install hydra-core
pip3 install pyrootutils
pip3 install scikit-learn
pip3 install pydantic

need huggingface login
=======
pip3 install openprompt torch hydra-core pyrootutils scikit-learn pydantic

<h1 align="center">
	🧑‍💻 CS BROKER
</h1>

<img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/SW13-Monstera/frontend&count_bg=%234E416D&title_bg=%23727272&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="hits" align='right' style='margin-left:5px;' />

<img src="https://img.shields.io/badge/version-v0.0.1-blue" alt="version0.0.1" align='right' style='margin-left:5px;'/>

🔗 배포: <https://csbroker.io>

<br/>

## 👋 소개

AI 기반 서술형 채점 기법을 통해
다양한 유형의 Computer Science 문제를 풀고
스스로 CS 지식을 학습할 수 있는 사이트입니다.

<br/>

## ✨ 기능

- ⭐ AI 기반 서술형 채점
- 📝 단답형, 객관식 문제 풀이
- 🙋‍♂️ 오늘의 문제 추천
- 📊 카테고리별 강약점 분석

<br/><br/>

## 🛠 채점 아이디어
### 키워드 채점
![BERT](https://user-images.githubusercontent.com/67869514/190990010-a9383189-dca3-4775-8784-029694253db1.png)

SBERT를 통해 Sliding Window 방식으로 키워드 여부를 확인합니다.


### 내용 채점
![T5](https://user-images.githubusercontent.com/67869514/190990325-9aafb37a-e123-46bb-abd6-97a69789270c.png)
![Prompt Tuning](https://user-images.githubusercontent.com/67869514/190990370-641ae66f-9638-46fb-aa0b-81fc1c99e832.png)

T5 모델을 Prompt Tuning 하여 사용합니다.

![image](https://user-images.githubusercontent.com/67869514/190989533-4b948c6f-79fa-481c-ab4a-06da4e80ff3d.png)

Prompt Tuning의 예시입니다.

### 정리된 슬라이드 : [Google Slide](https://docs.google.com/presentation/d/1TWTurKIexCGG0GJZRPvv_XQyipcFerA3x7cPF42fQ7o/edit#slide=id.g1567aa70b70_0_45)


<br/><br/>

## 📂 프로젝트 구조

```

📁 CS Broker (Ai-research)
├── README.md
├── pyproject.toml
├── requirements.txt
├──📁 core
├──📁 data_parsing_scripts
├──📁 datasets
├──📁 notebook
├──📁 prompt_tuning
│   ├── dataset.py
│   ├── loss.py
│   ├── train.py
│   ├── utils.py
│   ├──📁 outputs
│   └──📁 wandb
├──📁 configs
│   ├── main.yaml
│   ├──📁 dataset
│   ├──📁 model
│   └──📁 paths
└──📁 static

```

<br/><br/>

## 🤙🏻 협업 내용

- [그라운드 룰][ground-rule]
- [컨벤션][convention]
- [팀 노션][notion]
- [피그마][figma]



<br/><br/>

## 👩🏻‍💻 팀원

|                      **Kim-Hyunjo**                      |                      **kshired**                      |                      **ekzm8523**                      |
| :------------------------------------------------------: | :---------------------------------------------------: | :----------------------------------------------------: |
| <img src="https://github.com/Kim-Hyunjo.png" width="80"> | <img src="https://github.com/kshired.png" width="80"> | <img src="https://github.com/ekzm8523.png" width="80"> |
|         [김현조](https://github.com/Kim-Hyunjo)          |         [김성일](https://github.com/kshired)          |         [민재원](https://github.com/ekzm8523)          |

[ground-rule]: https://github.com/SW13-Monstera/.github/wiki/Ground-Rule
[convention]: https://github.com/SW13-Monstera/.github/wiki/Convention
[notion]: https://seed-cry-ce7.notion.site/QUARTER-f5f30a4b31264ae48129812cfb6e67f0
[figma]: https://www.figma.com/file/aBDgy14qYv8oEiqC6n8p4S/CS%2BBROKER-(1)?node-id=0%3A1
