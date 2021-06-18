import os

import boto3

file_name = f'/opt/ml/serving/models/model.pt'
bucket = 'boostcamp-dkt-data'
key = 'models/model.pt'

print('#'*30)
print(f"CURRENT WORKING DIRECTORY : {os.path.dirname(os.path.realpath(__file__))}")
print(f"UPLOAD DATA TO {bucket}/{key} FROM {file_name} ...")

s3 = boto3.client('s3')
res = s3.upload_file(file_name, bucket, key)

print(f"MODEL UPLOADED : [{file_name}]")
print('#'*30)
