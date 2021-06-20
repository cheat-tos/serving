from service import PytorchDKT
from args import parse_args

import os
import numpy as np
import pandas as pd

import torch
from dkt.trainer import get_model

# get argument from config.json
args = parse_args(mode='train')

# inference server setting
args.device = "cpu"
args.data_dir = "/root/serving/data/"
args.asset_dir = "/root/serving/asset/"
args.model_dir = "/root/serving/models/"
args.output_dir = "/root/serving/output/"

# add additional arguments
args.cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
args.cont_cols = []
args.features = []
args.n_cols = {}

args.cate_cols.extend(['paperID', 'head', 'mid', 'tail'])
args.cont_cols.append('Timestamp')
args.features.extend(
    ['answerCode'] + 
    args.cate_cols + 
    args.cont_cols
)

# load label encoder array
le = {}
for col in args.cate_cols:
    le[col] = np.load(os.path.join(args.asset_dir,col+'_classes.npy'))

# get addtitional arguments
for col in args.cate_cols:
    args.n_cols[col] = len(le[col])

# get trained model
args.model='saint'
model_path = os.path.join(args.model_dir, 'model.pt')
load_state = torch.load(model_path, map_location=torch.device(args.device))
model = get_model(args)
model.load_state_dict(load_state['state_dict'], strict=True)

# get test data
test = pd.read_csv("/root/serving/questions.csv")

# # for container-host volume mapping
# args.data_dir = "/home/bentoml/data/" # map to "/root/serving/data/"
# args.asset_dir = "/home/bentoml/asset/"# map to "/root/serving/asset/"
# args.model_dir = "/home/bentoml/models/" # map to "/root/serving/models/"
# args.output_dir = "/home/bentoml/output/" # map to "/root/serving/output/"

# packing
bento_dkt = PytorchDKT()
bento_dkt.pack('model', model)
bento_dkt.pack('test', test)
bento_dkt.pack('config', args)
for col in args.cate_cols:
    bento_dkt.pack(col+'_classes', le[col])

# save
saved_path = bento_dkt.save() # develop : in local / production : in object storage service
print(saved_path)
