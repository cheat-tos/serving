import bentoml
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact, JSONArtifact
from bentoml.adapters import JsonInput

from easydict import EasyDict
import time
from datetime import datetime
import random

import inference


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact('model'), PickleArtifact('test'), JSONArtifact('config'), \
    PickleArtifact('assessmentItemID_classes'), PickleArtifact('testId_classes'), PickleArtifact('KnowledgeTag_classes'), \
    PickleArtifact('paperID_classes'), PickleArtifact('head_classes'), PickleArtifact('mid_classes'), PickleArtifact('tail_classes')]) 
class PytorchDKT(bentoml.BentoService):
     
    #이번에는 input을 파일 자체로 입력받겠습니다.
    #마찬가지로 주소는 HOST:PORT/predict 입니다.
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, data):
        # get config
        args = EasyDict(self.artifacts.config)

        # get label encoder list
        le = {}
        le['assessmentItemID'] = self.artifacts.assessmentItemID_classes
        le['testId'] = self.artifacts.testId_classes
        le['KnowledgeTag'] = self.artifacts.KnowledgeTag_classes
        le['paperID'] = self.artifacts.paperID_classes
        le['head'] = self.artifacts.head_classes
        le['mid'] = self.artifacts.mid_classes
        le['tail'] = self.artifacts.tail_classes

        # transform data into input structure
        user_data = []
        for d in data:
            if 'answer' in d:
                row = [d['assess_id'], d['test_id'],d['tag'], d['timestamp'], d['answer']]
                user_data.append(row)

        # set data data`s timestamp randomly (10~15 sec)
        last_timestamp = user_data[-1][-2]
        last_sec = time.mktime(datetime.strptime(last_timestamp, '%Y-%m-%d %H:%M:%S').timetuple()) 
        sec_list = []
        for _ in range(len(self.artifacts.test)):
            sec_list.append(datetime.fromtimestamp(last_sec+random.randint(5,15)).strftime('%Y-%m-%d %H:%M:%S'))
        self.artifacts.test['Timestamp'] = sec_list

        
        print('* data :  \n', data)
        print('-'*100)
        print('* user data : \n', user_data)
        print('-'*100)
        print('* test data : \n', self.artifacts.test)
        print('-'*100)
        print('* config : \n', self.artifacts.config)
        print('-'*100)
        print('* args : \n', args)
        print('-'*100)
        print('* model : \n',  self.artifacts.model)
        print('-'*100)
        print('* assessmentItemID_classes : \n',  self.artifacts.assessmentItemID_classes)
        print('-'*100)
        
        score = inference.inference(user_data, self.artifacts.test, self.artifacts.model, le, args)

        print('* score : \n', score)

        return score
