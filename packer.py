from service import PytorchDKT
from args import parse_args

import os
import numpy as np
import pandas as pd

import torch
from dkt.trainer import get_model

# get argument from config.json
args = parse_args(mode='train')

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
model_path = os.path.join(args.model_dir, 'model.pt')
load_state = torch.load(model_path)
model = get_model(args)
model.load_state_dict(load_state['state_dict'], strict=True)

# get test data
test = pd.read_csv("questions.csv")

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
