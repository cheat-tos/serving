import os

import torch
import pandas as pd
import numpy as np

from args import parse_args
from dkt import trainer
from dkt.dataloader import Preprocess

"""
1. 모델은 전역변수로 로딩할것(모델은 inference 시 이미 메모리에 올라가있어야함)
2. label이나 모델 weight같은 파일들을 옮겨서 사용할것
"""
# args.embedding
def load_label_embedding(args):
    args.n_questions = len(np.load(os.path.join(args.asset_dir, 'assessmentItemID_classes.npy')))
    args.n_test = len(np.load(os.path.join(args.asset_dir, 'testId_classes.npy')))
    args.n_tag = len(np.load(os.path.join(args.asset_dir, 'KnowledgeTag_classes.npy')))

def gen_data(data):
    df = pd.read_csv("questions.csv") # 질문 목록

    new_columns = df.columns.tolist()+['answerCode'] # question.csv data frame에 정답여부 추가(마킹 위해)
    new_df = pd.DataFrame([],columns=new_columns+['userID']) # userID와 answerCode를 가지고 있는 빈 dataframe 생성
    
    for index, row in df.iterrows(): # 맞추어야하는 6n번째 question data(20개)
        # 5개의 (전처리된) 유저 입력 데이터를 new_df에 삽입
        user_actions = pd.DataFrame(data, columns=new_columns)
        user_actions['userID'] = index # 유저 구분하지는 않고 그냥 문제 푼 순서를 userID로 삼음
        new_df=new_df.append(user_actions)

        # n번째 질문을 6번째로 삽입
        row['userID'] = index
        new_df=new_df.append(row)
    
    new_df['answerCode'].fillna(-1, inplace=True) # 6번째 문항은 answerCode가 -1(맞추어야하는 문항)
    new_df['answerCode']=new_df['answerCode'].astype(int)
    new_df['KnowledgeTag']=new_df['KnowledgeTag'].astype(str)
    new_df = new_df.reset_index(drop=True) # 여기를 지나치기 전까지 index는 [0,1,2,3,4,0, 0,1,2,3,4,0,...]
    return new_df


#### global settings ####
args = parse_args(mode='inference') # inference
os.makedirs(args.model_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

load_label_embedding(args)
model = trainer.load_model(args) # 모델 전역변수 로딩

##########################
    
def inference(data):
    # RAW 데이터
    print("Before:",data) # [assessmentItemID, testId, KnowledgeTag, answer]
    # DATAFRAME 데이터
    data = gen_data(data)
    print("After:",data)
    # FE 및 PREPROCESSING 데이터
    preprocess = Preprocess(args)
    grouped_values = preprocess.preprocess_for_inference(data, is_train=False)
    
    #TODO
    #이곳에서 위에서 생성한 데이터를 기반으로 inference한 값들을 평균을 내서 입력해주시면 되겠습니다.
    # 예측해야할 (6n번째) 문제 값들의 answerCode 확률값(0~1.0) 평균 * 100
    pred_df = trainer.inference(args, model, grouped_values)

    result = pred_df['probability'].mean() * 100

    return result    

