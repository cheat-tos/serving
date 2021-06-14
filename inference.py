import torch
import pandas as pd
import numpy as np

import labeled_dataloader as dataloader


def process_batch(batch, args):

    # test, question, tag, correct, mask = batch
    correct, question, test, tag, paperid, head, mid, tail, time, mask = batch
    
    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    # #  interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    # #    saint의 경우 decoder에 들어가는 input이다
    # interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    # interaction = interaction.roll(shifts=1, dims=1)
    # interaction[:, 0] = 0 # set padding index to the first sequence
    # interaction = (interaction * mask).to(torch.int64)

    # print(interaction)
    # exit()
    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # 추가된 feature
    paperid = ((paperid + 1) * mask).to(torch.int64)
    head = ((head + 1) * mask).to(torch.int64)
    mid = ((mid + 1) * mask).to(torch.int64)
    tail = ((tail + 1) * mask).to(torch.int64)
    time = ((time + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동

    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    paperid = paperid.to(args.device)
    head = head.to(args.device)
    mid = mid.to(args.device)
    tail = tail.to(args.device)
    time = time.to(args.device)

    return (test, question,
            tag, correct, mask,
            interaction, paperid, head, mid, tail, time, gather_index)



def gen_data(data, test):
    df = test
    
    new_columns = df.columns.tolist()+['answerCode']
    new_df = pd.DataFrame([], columns=new_columns+['userID'])
    
    for index, row in df.iterrows():
        user_actions = pd.DataFrame(data, columns=new_columns)    
        user_actions['userID'] = index
        new_df=new_df.append(user_actions)
        row['userID'] = index
        new_df=new_df.append(row)
    
    new_df['answerCode'].fillna(-1, inplace=True)
    new_df['answerCode']=new_df['answerCode'].astype(int)
    new_df['KnowledgeTag']=new_df['KnowledgeTag'].astype(str)
    
    return new_df


def inference(data, test, model, le, args):
    
    data = gen_data(data, test)
    
    preprocess = dataloader.Preprocess(args, le)
    preprocess.load_test_data(data)
    test_data = preprocess.get_test_data()
    
    print('* test data : \n', test_data)
    print('-'*100)
    
    # inference
    model.eval()
    _, test_loader = dataloader.get_loaders(args, None, test_data)
    
    total_preds = []
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)

        preds = model(input)
        # predictions
        preds = preds[:,-1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    result = 100*sum(total_preds)/len(total_preds)
    
    return result    
