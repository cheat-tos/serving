import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import numpy as np

def evaluation(gt_path, pred_path):
    """
    Args:
        gt_path (string) : root directory of ground truth file
        pred_path (string) : root directory of prediction file (output of inference.py)
    """
    #어떤 gt를 사용하느냐에 따라 달라짐, 
    #제출 ID에서 gt에 있는 ID 값만 채점
    
    gt = pd.read_csv(gt_path, index_col='id')
    total_targets = gt['answerCode'].values

    pred = pd.read_csv(pred_path,index_col='id')
    #ground truth에 있는 id 값만 골라내기
    total_preds = pred.loc[list(gt.index),'prediction']

    # AUROC
    auroc = roc_auc_score(total_targets, total_preds)
    acc = accuracy_score(total_targets, np.where(total_preds >= 0.5, 1, 0))
    results={}
    results['accuracy'] = {
        'value': f'{acc:.4f}',
        'rank': False,
        'decs': True,
    }
    results['auroc'] = {
        'value': f'{auroc:.4f}',
        'rank': True,
        'decs': True,
    }

    return json.dumps(results)

# gt_path = 'data/eval_dataset/ground_truth/private_gt.csv'
# pred_path = 'data/eval_dataset/sample_submission.csv'
# print(evaluation(gt_path, pred_path))