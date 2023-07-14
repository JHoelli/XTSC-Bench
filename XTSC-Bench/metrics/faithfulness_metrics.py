import quantus
import numpy as np
import torch
import pandas as pd
from Benchmarking.metrics.metrics_helper import Quantus_Wrapper
from Benchmarking.metrics.synthetic_metrics import generateNewSample
import copy
from quantus.helpers.utils import (
    get_baseline_value,
    blur_at_indices,
    expand_indices,
    get_leftover_shape,
    offset_coordinates,
)
# USE Quantus

def time_series_pertubation(arr: np.array, **kwargs) -> np.array:
    """
    Time Series Specific Pertubation

    Parameters
    ----------
    arr: np.ndarray
         Array to be perturbed.
    kwargs: optional
        Keyword arguments.

    Returns
    -------
    arr: np.ndarray
         Array unperturbed.

    #TODO 
    """
    return arr

def uniform_noise_parameter():
    '''
        lower_bound: float
            The lower bound for uniform sampling.
    upper_bound: float, optional
            The upper bound for uniform sampling.
            '''
    pass

def syntheticBaseline(dataGenerationProcess, NumTimeSteps,NumFeatures, mode):
    sample= generateNewSample(dataGenerationProcess, sampler="irregular", NumTimeSteps= NumTimeSteps, NumFeatures=NumFeatures)
    #print('Sample', sample.shape)
    #if mode=='feat':
    return sample.reshape( sample.shape[-1], sample.shape[-2])
    #return sample
    

def baseline_replacement_by_indices(
    arr,
    indices,  # Alt. Union[int, Sequence[int], Tuple[np.array]],
    indexed_axes,
    perturb_baseline,
    **kwargs,
) -> np.array:
    """
    #TODO THIS NEEDS TO BE RECHECKED FOR MUTIVARIATE
    """
    #print('Indicies',indices)
    indices = expand_indices(arr, indices, indexed_axes)
    baseline_shape = get_leftover_shape(arr, indexed_axes)
    #print('Baselin SHape', baseline_shape)

    arr_perturbed = copy.copy(arr)

    # Get the baseline value.
    baseline_value = perturb_baseline[indices]
    #print(baseline_value)

    # Perturb the array.
    arr_perturbed[indices] = np.expand_dims(baseline_value, axis=tuple(indexed_axes))
    #print(arr_perturbed.shape)

    return arr_perturbed
      

#def perturb_baseline():
#    pass

def faithfulnessEstimate (mod,data,label, res,exp,perturb_baseline, perturb_func=quantus.perturb_func.baseline_replacement_by_indices):
    metric= quantus.FaithfulnessEstimate(
    perturb_func=perturb_func,
    similarity_func=quantus.similarity_func.correlation_pearson,
    perturb_baseline=perturb_baseline,disable_warnings=True)
    return  metric ( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=True)

def faithfulnessCorrelelation(mod,data,label, res,exp, perturb_baseline,perturb_func=quantus.perturb_func.baseline_replacement_by_indices,subset_size=30):
    metric= quantus.FaithfulnessCorrelation( nr_runs=100,    subset_size=subset_size,  perturb_func=perturb_func,
    similarity_func=quantus.similarity_func.correlation_pearson,  
    abs=False,  
    return_aggregate=False,
    perturb_baseline=perturb_baseline,disable_warnings=True)
    return  metric ( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=True)

def monoton(mod,data,label, res,exp,perturb_baseline='mean',perturb_func=quantus.perturb_func.baseline_replacement_by_indices):
    '''
     starts from a reference baseline to then incrementally replace each feature in a sorted attribution vector, measuring the effect on model performance 
    #TODO Parameters
    
    '''

    metrics = quantus.Monotonicity(  perturb_baseline=perturb_baseline, perturb_func=perturb_func,disable_warnings=True)
    scores = metrics( model=mod,  x_batch=data,y_batch=label, a_batch=res,  device='cpu',explain_func=exp,channel_first=True)
    return scores

def monotonicityCorrelation(mod,data,label, res,exp,perturb_baseline='mean',perturb_func=quantus.perturb_func.baseline_replacement_by_indices):
    '''
     starts from a reference baseline to then incrementally replace each feature in a sorted attribution vector, measuring the effect on model performance 
    #TODO Parameters
    
    '''
    # features_in_step=features_in_step,

    metrics =quantus.MonotonicityCorrelation( nr_samples=10,   perturb_baseline=perturb_baseline,  perturb_func=perturb_func,
    similarity_func=quantus.similarity_func.correlation_spearman,disable_warnings=True)
    scores = metrics( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=True)
    return scores



def quantus_faithfulness_wapper(metric, mod,data,label, res,exp,channel_first):
    #TODO
    return  metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp)

import torch
class model_wrapper(torch.nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model=model
    def forward(self,x):
        x=x.reshape(-1,x.shape[-1],x.shape[-2])
        return self.model(x)


def get_faithfullness_metrics( original,exp,mlmodel,labels=None,explainer=None,mode='time', additional_metrics=None,generation_process= None,subset_size=30):
    '''
    Calculated the Faithfullness Metrics on quantus Basis. 

    Attributes: 
        original: 
        exp: 
        mlmodel: 
        labels: 
        explainer: 
        mode: 
        additional_metrics:
        generation_process:
    '''

    if mode== 'time':
        channel_first=False
        num_feat= original.shape[-1]
        num_time= original.shape[-2]
        original=original.reshape(-1, num_feat,num_time)
        exp=exp.reshape(-1, num_feat,num_time)
        mlmodel=model_wrapper(mlmodel)
        mlmodel.eval()
    else: 
        channel_first=True
        num_feat= original.shape[-2]
        num_time= original.shape[-1]

    explainer=Quantus_Wrapper(explainer).make_callable
    df = pd.DataFrame([])
    if num_feat<2:
        #TODO IF AS PARAM
        for a in ["uniform","mean"]:
            #TODO Neighbor hood mean, neighbor hood min max
            try:
                df[f'monoton_{a}']= np.array(monoton(mlmodel,original,labels, exp,explainer,perturb_baseline=a))
            except:
                df[f'monoton_{a}']=  np.array(np.repeat(np.nan,len(original)))
            try:
                df[f'faithfulness_correlation_{a}']=  np.array(faithfulnessCorrelelation(mlmodel,original,labels, exp,explainer,perturb_baseline=a, subset_size=subset_size))
            except:
            
                df[f'faithfulness_correlation_{a}']=  np.array(np.repeat(np.nan,len(original)))
        
    if generation_process is not None:
        baseline = syntheticBaseline(generation_process, num_time, num_feat, mode)
        try:
            df[f'monoton_synthetic']= np.array(monoton(mlmodel,original,labels, exp,explainer,baseline,baseline_replacement_by_indices))
        except:
            df[f'monoton_synthtic']=  np.array(np.repeat(np.nan,len(original)))
        try:
            df[f'faithfulness_correlation_synthetic_flex']=  np.array(faithfulnessCorrelelation(mlmodel,original,labels, exp,explainer,baseline,baseline_replacement_by_indices,subset_size=subset_size))
        except:
            df[f'faithfulness_correlation_synthtic_flex']=  np.array(np.repeat(np.nan,len(original)))
        #df[f'faithfulness_correlation_synthetic_flex']=  np.array(faithfulnessCorrelelation(mlmodel,original,labels, exp,explainer,baseline,baseline_replacement_by_indices,subset_size=subset_size))

    
    if additional_metrics is not None: 
        for add in additional_metrics:
            df[f'{str(type(add))}']= np.array(quantus_faithfulness_wapper(mlmodel,original,labels, exp,explainer))
    #print(df)
    #import sys 
    #sys.exit(1)

    return df
