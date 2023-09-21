import quantus
import numpy as np
import torch
import pandas as pd
from XTSCBench.metrics.metrics_helper import Quantus_Wrapper
from XTSCBench.metrics.synthetic_metrics import get_reference_samples




def quantus_localization_wapper(metric,mlmodel,data,labels, exp,masks):
    return  metric( model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU')


def get_reliability_metrics( data,exp,mlmodel,labels,meta, shape, mode='time', additional_metrics=None, synthtic=True):
    print('data ',data.shape)
    print('exp ',exp.shape)
    print(meta)
    labels=labels.astype(int)
    df = pd.DataFrame([])

    if np.all((exp== 0.0)):
        #print('IF')
        df['Pointing']=[np.nan]
        df['Relevance Rank']=[np.nan]
        df['Relevance Mass']=[np.nan]
        df['AuC']=[np.nan]
        return df

    exp= exp.astype(np.float64)
    if synthtic:
        #TODO
        masks=get_reference_samples(meta,data,shape)#.reshape(data.shape[0],data.shape[2],data.shape[1])
        if mode =='feat':
            masks=np.swapaxes(masks, -1,-2)
        print('masks ',masks.shape)
    else:
        masks=meta

    exp=np.array(exp).reshape(data.shape[0],1,data.shape[2]*data.shape[1])
    masks=np.array(masks).reshape(data.shape[0],1,data.shape[2]*data.shape[1])
    data=np.array(data).reshape(data.shape[0],1,data.shape[2]*data.shape[1])

    pointing=quantus.PointingGame(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU')
    rank=quantus.RelevanceRankAccuracy(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU')
    rank_mass=quantus.RelevanceMassAccuracy(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU')
    try:
        Auc=quantus.AUC(disable_warnings=True)(model=mlmodel,x_batch=data, y_batch=labels,  a_batch=exp, s_batch=masks, device='CPU')
    except: 
        Auc=np.repeat(np.nan,len(rank))
    if additional_metrics is not None: 
        for add in additional_metrics:
            df[f'{str(add)}']= np.array(quantus_localization_wapper(add,mlmodel,data,labels, exp,masks))
    if np.all((exp==0.0)):
        return pd.DataFrame([np.nan,np.nan, np.nan],columns=["Poining", "Relevance Rank","Relevance Mass","AuC"])
    df['Pointing']=pointing
    df['Relevance Rank']=rank
    df['Relevance Mass']=rank_mass
    df['AuC']=Auc
    return df
   
    
