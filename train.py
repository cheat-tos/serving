import os
import torch
from dkt import trainer
from dkt.utils import setSeeds
from dkt.dataloader import Preprocess
from args import parse_args

import boto3
import pandas as pd

# import wandb
def main(args):
    # wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    
    train_data, valid_data = preprocess.split_data(train_data)
    
    # wandb.init(project='', entity='', config=vars(args))
    # wandb.run.name = args.config
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')

    # load training data from s3
    os.makedirs(args.data_dir, exist_ok=True)
    file_name = args.data_dir+'train_data.csv'
    bucket = 'boostcamp-dkt-data'
    key = 'train_dataset/train_data.csv'
    client = boto3.client('s3') 
    client.download_file(bucket, key, file_name)
    # args.data_dir = '/opt/ml/input/data/train_dataset'

    # set df for additional data
    additional_df = pd.DataFrame([], columns=['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp', 'KnowledgeTag'])
    additional_df.to_csv(args.data_dir+'additional_data.csv', index=False)

    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
