import os
from datetime import datetime
import time
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

class Preprocess:
    def __init__(self, args, le=None):
        self.args = args
        self.train_data = None
        self.test_data = None

        self.args.cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
        self.args.cont_cols = []
        self.args.features = []
        self.args.n_cols = {}

        self.le = le

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        cate_cols = self.args.cate_cols

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                le.classes_ = self.le[col]
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
        
        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df):
        # user split by seq_len
        seq_len = self.args.max_seq_len
        new_id = 0
        before = 0
        count = 0
        new_user = []

        userid = df.userID.tolist()
        # userid.reverse()

        for u in userid:
            if (count == seq_len) or (u != before):
                new_id += 1
                count = 0
                
            new_user.append(new_id)
            count += 1
            
            before = u
        
        # new_user.reverse()
        # max_user = max(new_user)
        # new_user = [max_user - n for n in new_user]
        
        df['newID'] = new_user
        
        df['paperID'] = df.assessmentItemID.apply(lambda x: x[1:7])
        df['head'] = df.assessmentItemID.apply(lambda x: x[1:4])
        df['mid'] = df.assessmentItemID.apply(lambda x: x[4:7])
        df['tail'] = df.assessmentItemID.apply(lambda x: x[7:])

        self.args.cate_cols.extend(['paperID', 'head', 'mid', 'tail'])
        self.args.cont_cols.append('Timestamp')

        self.args.features.extend(
            ['answerCode'] + 
            self.args.cate_cols + 
            self.args.cont_cols
            )

        return df

    def load_data_from_file(self, file_name):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)#, nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train=True)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        for col in self.args.cate_cols:
            self.args.n_cols[col] = len(np.load(os.path.join(self.args.asset_dir, f'{col}_classes.npy')))
    
        for col in self.args.cont_cols:
            self.args.n_cols[col] = len(df[col].unique()) + 1


        df = df.sort_values(by=[self.args.data_id,'Timestamp'], axis=0)
        
        columns = self.args.features

        def get_values(cols, r):
            result = []
            for col in cols:
                result.append(r[col].values)

            return result

        group = df[[self.args.data_id] + columns].groupby(self.args.data_id).apply(
                lambda r: (
                    get_values(columns, r)
                )
            )

        return group.values

    def load_data_from_df(self, df):
        
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train=False)
        
        def get_values(cols, r):
            result = []
            for col in cols:
                result.append(r[col].values)
            return result
        
        columns = columns = self.args.features
        group = df[['userID'] + columns].groupby('userID').apply(
            lambda r: (
                get_values(columns, r)
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, data):
        self.test_data = self.load_data_from_df(data)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        # test, question, tag, correct = row[0], row[1], row[2], row[3]
        # cate_cols = [test, question, tag, correct]

        cols = list(row)

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cols):
                cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cols):
            cols[i] = torch.tensor(col)

        return cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)  # padding을 앞에 추가?


    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = True
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=0, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader