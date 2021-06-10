import bentoml
from bentoml import artifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.artifact import PickleArtifact, JSONArtifact
from bentoml.adapters import JsonInput

from easydict import EasyDict
import inference


@bentoml.env(infer_pip_packages=True)  #(requirements_txt_file="./requirements.txt")
@bentoml.artifacts([PytorchModelArtifact('model'), PickleArtifact('test'), JSONArtifact('config'), \
    PickleArtifact('assessmentItemID_classes'), PickleArtifact('testId_classes'), PickleArtifact('KnowledgeTag_classes')]) 
class PytorchDKT(bentoml.BentoService):
     
    #이번에는 input을 파일 자체로 입력받겠습니다.
    #마찬가지로 주소는 HOST:PORT/predict 입니다.
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, data):
        args = EasyDict(self.artifacts.config)

        le = {}
        le['assessmentItemID'] = self.artifacts.assessmentItemID_classes
        le['testId'] = self.artifacts.testId_classes
        le['KnowledgeTag'] = self.artifacts.KnowledgeTag_classes

        user_data = []
        for d in data:
            if 'answer' in d:
                row = [d['assess_id'], d['test_id'],d['tag'], d['answer']]
                user_data.append(row)

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
        