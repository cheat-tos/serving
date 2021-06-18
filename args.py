import os
import argparse
import json
from easydict import EasyDict

def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='lstm')
    args_config = parser.parse_args()

    args = EasyDict()
    with open(f'/root/serving/config/{args_config.config}.json', 'r') as f:
        args.update(json.load(f))

    args['config'] = args_config.config

    return args
