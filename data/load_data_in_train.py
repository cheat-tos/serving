import os
import pandas as pd
import boto3

# load training data from s3
file_name = '/opt/ml/serving/data/train_data.csv'
bucket = 'boostcamp-dkt-data'
key = 'train_dataset/train_data.csv'

print('#'*30)
print(f"CURRENT WORKING DIRECTORY : {os.path.dirname(os.path.realpath(__file__))}")
print(f"LOADING DATA FROM {bucket}/{key} AS {file_name} ...")

client = boto3.client('s3')
client.download_file(bucket, key, file_name)
