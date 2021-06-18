import pandas as pd
import boto3

# load training data from s3
file_name = 'data/train_data.csv'
bucket = 'boostcamp-dkt-data'
key = 'train_dataset/train_data.csv'
client = boto3.client('s3') 
client.download_file(bucket, key, file_name)

# set df for additional data
additional_df = pd.DataFrame([], columns=['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp', 'KnowledgeTag'])
additional_df.to_csv('data/additional_data.csv', index=False)
