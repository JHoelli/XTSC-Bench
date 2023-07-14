import quantus
import numpy as np
import torch
import pandas as pd
# USE Quantus

class Quantus_Wrapper():

    def __init__(self,explainer) :
        self.explainer=explainer
    def make_callable(self,model,inputs,targets, device='cpu'):
        self.explainer.model=model
        res=[]
        for x1, y1 in zip(inputs,targets):
            res.append(self.explainer.explain(x1.reshape(-1,inputs.shape[-2],inputs.shape[-1]), int(y1)))
        #print('res',np.array(res).shape)               
        return res

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

def similarity_function():
    pass
      
def MaxSensitivity(mod,data,label, res, exp,channel_first):
    '''
    measures the maximum sensitivity of an explanation using a Monte Carlo sampling-based approximation 
    '''

    metric = quantus.MaxSensitivity(nr_samples=1000,lower_bound=0.2, perturb_func=quantus.uniform_noise, similarity_func=quantus.difference)

    scores = metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)
    return scores

#def local_lipshits(mod,data,label, res,exp,channel_first):
    ''' tests the consistency in the explanation between adjacent examples '''
#    metric = quantus.LocalLipschitzEstimate( nr_samples=10,
#    perturb_std=0.2,
#    perturb_mean=0.0,
#    norm_numerator=quantus.similarity_func.distance_euclidean,
#    norm_denominator=quantus.similarity_func.distance_euclidean,    
#    perturb_func=quantus.perturb_func.gaussian_noise,
#    similarity_func=quantus.similarity_func.lipschitz_constant,)
#    #print(channel_first)
#    scores = metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)
#    return scores

def AverageSensitivity(mod,data,label, res, exp,channel_first):
    '''
    measures the average sensitivity of an explanation using a Monte Carlo sampling-based approximation 
    '''

    metric = quantus.AvgSensitivity(nr_samples=1000,lower_bound=0.2, perturb_func=quantus.uniform_noise, similarity_func=quantus.difference)

    scores = metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)
    return scores

class Man_model(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model= model

    def forward(self,input):
        input = input.reshape(-1,input.shape[-2],input.shape[-1])
        return self.model(input)

#def infidelity(mod,data,label, res,exp,channel_first):
    '''
    represents the expected mean square error between 1) a dot product of an attribution and input perturbation and 2) difference in model output after significant perturbation 
    '''
    #https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/quantus/functions/perturb_func.py
    #baseline_replacement_by_indices
#    mod=Man_model(mod)
#    mod.eval()
    
#    metric=quantus.Infidelity(perturb_baseline="uniform",
#    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
#    n_perturb_samples=5,
#    perturb_patch_sizes=[56], 
#    display_progressbar=True)
#    return metric ( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp, channel_first=channel_first)

def faithfulnessEstimate (mod,data,label, res,exp,channel_first,perturb_baseline):
    metric= quantus.FaithfulnessEstimate(
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_pearson,
    perturb_baseline=perturb_baseline,)
    return  metric ( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp, channel_first=channel_first)

def faithfulnessCorrelelation(mod,data,label, res,exp,channel_first,perturb_baseline):
    metric= quantus.FaithfulnessCorrelation( nr_runs=100,    subset_size=100, 
    perturb_baseline=perturb_baseline,
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_pearson,  
    abs=False,  
    return_aggregate=False)
    return  metric ( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp, channel_first=channel_first)

def monoton(mod,data,label, res,exp,features_in_step,perturb_baseline='mean'):
    '''
     starts from a reference baseline to then incrementally replace each feature in a sorted attribution vector, measuring the effect on model performance 
    #TODO Parameters
    
    '''
    features_in_step=features_in_step

    metrics = quantus.Monotonicity(  perturb_baseline=perturb_baseline, perturb_func=quantus.perturb_func.baseline_replacement_by_indices)
    scores = metrics( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp)
    return scores

def monotonicityCorrelation(mod,data,label, res,exp,features_in_step,perturb_baseline='mean'):
    '''
     starts from a reference baseline to then incrementally replace each feature in a sorted attribution vector, measuring the effect on model performance 
    #TODO Parameters
    
    '''
    # features_in_step=features_in_step,

    metrics =quantus.MonotonicityCorrelation( nr_samples=10,   perturb_baseline=perturb_baseline,  perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_spearman)
    scores = metrics( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp)
    return scores

#def sufficiency():
#    '''TODO'''
#    quantus.Sufficiency( threshold=0.6,  return_aggregate=False,)

#def selectivity (mod,data,label, res,exp,features_in_step,channel_first):
#    '''
#    measures how quickly an evaluated prediction function starts to drop when removing features with the highest attributed values 
#    '''
#    metrics = quantus.Monotonicity(features_in_step=features_in_step,  perturb_baseline="black", perturb_func=quantus.perturb_func.baseline_replacement_by_indices)
#    scores = metrics( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)
#    return scores

def complexity(mod,data,label, res,exp):
    metrics= quantus.Complexity()
    return metrics( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp)



def quantus_wapper(metric, mod,data,label, res,exp,channel_first):
    return  metric( model=mod,  x_batch=data,y_batch=label, a_batch=res, device='cpu',explain_func=exp,channel_first=channel_first)


def get_general_metrics( original,exp,mlmodel,labels=None,explainer=None,mode='time', additional_metrix=None):
    #TODO Put in CORRECT MODE
    if mode== 'time':
        #TODO SHOULD BE 1
        channel_first=False
        #num_feat= original.shape[-2]
    else: 
        channel_first=True
        #num_feat= original.shape[-2]
    #print(num_feat)
    explainer=Quantus_Wrapper(explainer).make_callable
    df = pd.DataFrame([])
   
    #df['faithfulness_estimate']=  np.array(faithfulnessEstimate(mlmodel,original,labels, exp,explainer,channel_first))
    #df['LocalLipshitz']=np.array(local_lipshits(mlmodel,original,labels, exp,explainer,channel_first))
    #df['selectivity']= np.array(selectivity(mlmodel,original,labels, exp,explainer,channel_first,10))
    for a in ["random","mean"]:
        df[f'monoton_{a}']= np.array(monoton(mlmodel,original,labels, exp,explainer,10,a))
        df[f'monoton_corr_{a}']= np.array(monotonicityCorrelation(mlmodel,original,labels, exp,explainer,10,a))
        df[f'faithfulness_correlation_{a}']=  np.array(faithfulnessCorrelelation(mlmodel,original,labels, exp,explainer,channel_first,a))
    #TODO add Monotocity correlation
    df['complexity']= np.array(complexity(mlmodel,original,labels, exp,explainer))
    df['AverageSensitivity']=np.array(AverageSensitivity(mlmodel,original,labels, exp,explainer,channel_first))
    df['MaxSensitivity']=np.array(MaxSensitivity(mlmodel,original,labels, exp,explainer,channel_first))
    
    if additional_metrix is not None: 
        for add in additional_metrix:
            df[f'{str(type(add))}']= np.array(quantus_wapper(mlmodel,original,labels, exp,explainer))

    return df
