import quantus
import numpy as np
import torch
import pandas as pd
from XTSCBench.metrics.metrics_helper import Quantus_Wrapper


def complexity(mod,data,label, res,exp):
    if np.all((res==0.0)):
        return np.nan
    metrics= quantus.Complexity(disable_warnings=True)
    return metrics( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp)



def quantus_complexity_wrapper(metric, mod,data,label, res,exp):
    #TODO Add KWARGS Possibility
    return  metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp)


def get_complexity_metrics( original,exp,mlmodel,labels=None,explainer=None,mode='time', additional_metrics=None):
    original= original.reshape(-1,1,original.shape[-2]*original.shape[-1])
    exp= exp.reshape(-1,1,exp.shape[-2]*exp.shape[-1])

    explainer=Quantus_Wrapper(explainer).make_callable

    df = pd.DataFrame([])
    comp=complexity(mlmodel,original,labels, exp,explainer)
    df['complexity']= np.array(comp)

    if additional_metrics is not None: 
        for add in additional_metrics:
            #metric, mod,data,label, res,exp)
            df[f'{str(add)}']= np.array(quantus_complexity_wrapper(add,mlmodel,original,labels, exp,explainer))

    return df
