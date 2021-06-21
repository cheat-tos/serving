import os
import argparse
import json
from pathlib import Path

from easydict import EasyDict

base_path = Path(__file__).parent.absolute() # absolute path to project dir

def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='lstm')
    args_config = parser.parse_args()

    args = EasyDict()
    with open(f'{base_path}/config/{args_config.config}.json', 'r') as f:
    #with open(f'/root/serving/config/{args_config.config}.json', 'r') as f:
        args.update(json.load(f))

    args['config'] = args_config.config

    return args
