import pandas as pd
import boto3

train_data = pd.read_csv('train_data.csv')
additional_data = pd.read_csv('additional_data.csv')
additional_data.userID = additional_data.userID+list(train_data.userID)[-1]+1

added_train_df = pd.concat([train_data, additional_data], axis=0)
added_train_df.to_csv('added_train_data.csv', index=False)

file_name = 'added_train_data.csv'
bucket = 'boostcamp-dkt-data'
key = 'train_dataset/train_data.csv'
s3 = boto3.client('s3') 
res = s3.upload_file(file_name, bucket, key)