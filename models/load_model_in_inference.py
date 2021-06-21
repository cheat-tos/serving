import os

import boto3

file_name = f'/root/serving/models/model.pt'
bucket = 'boostcamp-dkt-data'
key = 'models/model.pt'

print('#'*30)
print(f"CURRENT WORKING DIRECTORY : {os.path.dirname(os.path.realpath(__file__))}")
print(f"LOAD DATA FROM {bucket}/{key} TO {file_name} ...")

s3 = boto3.client('s3')
res = s3.download_file(bucket, key, file_name)

print(f"MODEL LOADED : [{file_name}]")
print('#'*30)
