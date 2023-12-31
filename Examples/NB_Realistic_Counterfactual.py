
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
import torch 
from XTSCBench.ClassificationModels.CNN_T import ResNetBaseline, UCRDataset,fit
from XTSCBench.ClassificationModels.LSTM import LSTM
from XTSCBench.CounterfactualEvaluation import CounterfactualEvaluation
from tslearn.datasets import UCR_UEA_datasets
import sklearn
import numpy as np 
import os

dataset='ECG5000'
#For use with CNN reverse Data Dimensions
train_x,train_y, test_x, test_y=UCR_UEA_datasets().load_dataset(dataset)

enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(np.vstack((train_y.reshape(-1,1),test_y.reshape(-1,1))))
train_y=enc1.transform(train_y.reshape(-1,1))
test_y=enc1.transform(test_y.reshape(-1,1))    

n_pred_classes =train_y.shape[1]
train_dataset = UCRDataset(train_x.astype(np.float64),train_y.astype(np.int64))
test_dataset = UCRDataset(test_x.astype(np.float64),test_y.astype(np.int64))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
if os.path.isfile('./Examples/temp_lstm'):
    model =LSTM(1, 10 ,n_pred_classes,rnndropout=0.1).to('cpu') 
    model = torch.load( './Examples/temp_lstm')
else:
    model =LSTM(1, 10 ,n_pred_classes,rnndropout=0.1).to('cpu') 
    fit(model,train_loader,test_loader)
    torch.save(model, './Examples/temp_lstm')
model.eval()
# For use with CNN set mode ='feat'
explainer =  [TSEvo(model= model,data=(train_x,train_y), mode = 'time',backend='PYT',epochs=10)]
bm=CounterfactualEvaluation(explainer=explainer)
#tems,label,model, exp='None', mode='time', aggregate=False
print(bm.evaluate(test_x[0:2], np.argmax(test_y[0:2],axis=1),model, mode='time',aggregate=True))