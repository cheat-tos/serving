import os
import torch
from dkt import trainer
from dkt.utils import setSeeds
from dkt.dataloader import Preprocess
from args import parse_args




# import wandb
def main(args):
    # wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    print(f"MODEL : {args.config} - {args.device}")

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)

    # wandb.init(project='', entity='', config=vars(args))
    # wandb.run.name = args.config
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
