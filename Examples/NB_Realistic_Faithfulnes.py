
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
import torch 
from XTSCBench.ClassificationModels.CNN_T import ResNetBaseline, UCRDataset,fit
from XTSCBench.ClassificationModels.LSTM import LSTM
from XTSCBench.FaithfulnessEvaluation import FaithfulnessEvaluation
from tslearn.datasets import UCR_UEA_datasets
import sklearn
import numpy as np 
import os

dataset='ECG5000'
#For use with CNN reverse Data Dimensions
train_x,train_y, test_x, y=UCR_UEA_datasets().load_dataset(dataset)

enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(np.vstack((train_y.reshape(-1,1),y.reshape(-1,1))))
train_y=enc1.transform(train_y.reshape(-1,1))
test_y=enc1.transform(y.reshape(-1,1))    

n_pred_classes =train_y.shape[1]
train_dataset = UCRDataset(train_x.astype(np.float64),train_y.astype(np.int64))
test_dataset = UCRDataset(test_x.astype(np.float64),y.astype(np.int64))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
if os.path.isfile( './Examples/temp_lstm'):
    model=torch.load( './Examples/temp_lstm')
else:
    model =LSTM(1, 10 ,n_pred_classes,rnndropout=0.1).to('cpu') 
    fit(model,train_loader,test_loader)
    torch.save(model,  './Examples/temp_lstm')
model.eval()

import importlib
import XTSCBench
importlib.reload(XTSCBench)
# For use with CNN set mode ='feat'
explainer =  [TSEvo(model= model,data=(train_x,train_y), mode = 'time',backend='PYT',epochs=10),Saliency_PTY(model, 140,1, method='GRAD', mode='time', tsr=True)]
bm=FaithfulnessEvaluation(explainer=explainer,mlmodel=None)
print(bm.evaluate(test_x[0:2], y[0:2],model,exp=None, mode='time',aggregate=True))