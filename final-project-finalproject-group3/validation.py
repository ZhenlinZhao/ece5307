# this validation file computes AUC, not accuracy!

import argparse
import numpy as np
import pickle
import bz2
import pandas as pd
from sklearn.metrics import roc_auc_score

# build parser 
parser = argparse.ArgumentParser(description='Validate a trained model for ece5307 project')
parser.add_argument("model_path",help="path to model file")
parser.add_argument("--Xtr_path",default="Xtr.csv",help="path to training-feature file")
parser.add_argument("--ytr_path",default="ytr.csv",help="path to training-label file")
parser.add_argument("--Xts_path",default="Xts.csv",help="path to test-feature file")
parser.add_argument("--yts_hat_path",default="yts_hat.csv",help="path to test-label-prediction file")

# parse input arguments
args = parser.parse_args()
model_path = args.model_path
Xtr_path = args.Xtr_path
ytr_path = args.ytr_path
Xts_path = args.Xts_path
yts_hat_path = args.yts_hat_path
if not yts_hat_path.endswith('.csv'): 
    print("Error: Argument of --yts_hat_path must end in .csv")
    quit()

# load data
Xtr = np.loadtxt(Xtr_path, delimiter=",")
ytr = np.loadtxt(ytr_path, delimiter=",")
Xts = np.loadtxt(Xts_path, delimiter=",")

# load model
if model_path.endswith('.json'):
    # XGBOOST
    from xgboost import XGBClassifier 
    model = XGBClassifier()
    model.load_model(model_path)
    ytr_hat = model.predict(Xtr,output_margin=True)
    yts_hat = model.predict(Xts,output_margin=True)
elif model_path.endswith('.pth'):
    # PYTORCH
    import torch 
    model = torch.jit.load(model_path)
    with torch.no_grad():
        ytr_hat = model(torch.Tensor(Xtr)).data.detach().numpy().ravel()
        yts_hat = model(torch.Tensor(Xts)).data.detach().numpy().ravel()
elif model_path.endswith('.pkl'): 
    # SKLEARN (DEPRECATED)
    model = pickle.load(open(model_path, 'rb'))
    name = type(model).__name__
    if name in ['RandomForestClassifier','ExtraTreesClassifier','VotingClassifier']:
        ytr_hat = model.predict_proba(Xtr)[:,1]
        yts_hat = model.predict_proba(Xts)[:,1]
    elif name in ['LogisticRegression','LinearSVC','SVC','NuSVC','RidgeClassifier','AdaBoostClassifier','BaggingClassifier','GradientBoostingClassifier','StackingClassifier','HistGradientBoostingClassifier']:
        ytr_hat = model.decision_function(Xtr)
        yts_hat = model.decision_function(Xts)
    else:
        raise ValueError('model type '+name+' not supported')
elif model_path.endswith('.bz2'): 
    # SKLEARN
    with bz2.BZ2File(model_path,'r') as f:
        model = pickle.load(f)
    name = type(model).__name__
    if name in ['RandomForestClassifier','ExtraTreesClassifier','VotingClassifier']:
        ytr_hat = model.predict_proba(Xtr)[:,1]
        yts_hat = model.predict_proba(Xts)[:,1]
    elif name in ['LogisticRegression','LinearSVC','SVC','NuSVC','RidgeClassifier','AdaBoostClassifier','BaggingClassifier','GradientBoostingClassifier','StackingClassifier','HistGradientBoostingClassifier']:
        ytr_hat = model.decision_function(Xtr)
        yts_hat = model.decision_function(Xts)
    else:
        raise ValueError('model type '+name+' not supported')
else:
    print("Error: Unrecognized extension on model_path.  Should be .pkl for Sklearn models, or .pth for PyTorch models, or .json for XGBoost models")
    quit()


# print training auc 
auc = roc_auc_score(ytr,ytr_hat)
print('training auc = ',auc)

# save test-label confidences in a csv file 
df = pd.DataFrame(data={'Id':np.arange(len(yts_hat)),
                        'Label':yts_hat}) # note lack of "np.int64()" !!
df.to_csv(yts_hat_path, index=False)
print('test label confidences saved in',yts_hat_path)
