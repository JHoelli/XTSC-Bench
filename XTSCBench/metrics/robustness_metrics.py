import quantus
import numpy as np
import torch
import pandas as pd
from XTSCBench.metrics.metrics_helper import Quantus_Wrapper
# USE Quantus


def MaxSensitivity(mod,data,label, res, exp,channel_first):
    '''
    measures the maximum sensitivity of an explanation using a Monte Carlo sampling-based approximation 
    '''

    metric = quantus.MaxSensitivity(nr_samples=20,lower_bound=0.2, perturb_func=quantus.uniform_noise, similarity_func=quantus.difference,disable_warnings=True)

    scores = metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)
    return scores


def AverageSensitivity(mod,data,label, res, exp,channel_first):
    '''
    measures the average sensitivity of an explanation using a Monte Carlo sampling-based approximation 
    '''
    #print('LABEL ',label)
    #print('Res ',res)
    #print('DATA ', data)

    metric = quantus.AvgSensitivity(nr_samples=20,lower_bound=0.2, perturb_func=quantus.uniform_noise, similarity_func=quantus.difference,disable_warnings=True)

    scores = metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)
    return scores



def quantus_roboustness_wrapper(metric, mod,data,label, res,exp,channel_first):
    ''' # TODO Specify'''
    return  metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)

import torch
class model_wrapper(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model=model
    def forward(self,x):
        print(x.shape)
        x=x.reshape(-1,x.shape[-1],x.shape[-2])
        return self.model(x)



def get_robustness_metrics( original,exp,mlmodel,labels=None,explainer=None,mode='time', additional_metrics=None):
    #TODO Put in CORRECT MODE
    exp= np.array(exp)
    original=np.array(original)
    #print('Relevant SHapes')
    #print(exp.shape)
    #print(original.shape)
    if mode== 'time': #and not synthetic:
        channel_first=True
        num_feat= original.shape[-1]
        num_time= original.shape[-2]
        original=np.swapaxes(original,-1,-2).reshape(-1, num_feat,num_time)
        exp=exp.reshape(-1, num_feat,num_time)
        #mlmodel=model_wrapper(mlmodel)
        #mlmodel.eval()
    else: 
        channel_first=True
        num_feat= original.shape[-2]
        num_time= original.shape[-1]
    #labels=labels.astype(int)
    explainer=Quantus_Wrapper(explainer, mode).make_callable
    df = pd.DataFrame([])
    #print('Labels ', labels)
    #try:
    df['AverageSensitivity']=np.array(AverageSensitivity(mlmodel,original,labels, exp,explainer,channel_first))
    #except: 
    #    df['AverageSensitivity']= np.array(np.repeat(np.nan,len(original)))
    #try:
    df['MaxSensitivity']=np.array(MaxSensitivity(mlmodel,original,labels, exp,explainer,channel_first))
    #except: 
    #     df['MaxSensitivity']=np.array(np.repeat(np.nan,len(original)))
    
    if additional_metrics is not None: 
        for add in additional_metrics:
            df[f'{str(type(add))}']= np.array(quantus_roboustness_wrapper(mlmodel,original,labels, exp,explainer))

    return df
