'''
This section and calculations are based on :
Pawelczyk, Martin, et al. "Carla: a python library to benchmark algorithmic recourse and counterfactual explanation algorithms." 
arXiv preprint arXiv:2108.00783 (2021).
'''
from cProfile import label
import os
from tslearn.neighbors import KNeighborsTimeSeries
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from scipy.spatial import distance
import pickle
from typing import List
from deap import base
from deap import creator
import torch
from XTSCBench.Evaluation import Evaluation
from XTSCBench.metrics.synthetic_helper import load_synthetic_data, manipulate_exp_method
from XTSCBench.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from XTSCBench.metrics.synthetic_helper import load_synthetic_data,manipulate_exp_method,scaling, get_explanation,does_entry_already_exist
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import torch.utils.data as data_utils

# TODO Exclude Mode

def distance_1(original: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """
    # TODO Axis = 1? 
    """
    #print('Original', original.shape)
    #print('Counterfactual', counterfactual.shape)
    le=original.shape[-1]*original.shape[-2]
    num= counterfactual.shape[0]
    delta=original.reshape(num,-1)-counterfactual.reshape(num,-1)
    # compute elements which are greater than 0
    return np.sum(delta != 0, dtype=np.float,axis=1)/le


def distance_2(original: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """
    """
    le=original.shape[-1]*original.shape[-2]
    num= counterfactual.shape[0]
    delta=original.reshape(num,-1)-counterfactual.reshape(num,-1)
    return np.sum(np.abs(delta), dtype=np.float,axis=1)/le


def distance_3(original: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """

    """
    num= counterfactual.shape[0]
    delta=original.reshape(num,-1)-counterfactual.reshape(num,-1)
    return np.sum(np.square(np.abs(delta)), dtype=np.float,axis=1).tolist()


def distance_4(original: np.ndarray,counterfactual: np.ndarray) -> List[float]:
    """
    """
    num= counterfactual.shape[0]
    delta=original.reshape(num,-1)-counterfactual.reshape(num,-1)
    return np.max(np.abs(delta),axis=1).tolist()

def yNN(
    counterfactuals,
    mlmodel,data,
    y,labels=None
) :
    """    
    """
    number_of_diff_labels = 0
    if labels==None:
        labels = [np.argmax(cf.output) for cf in counterfactuals]
    


    counterfactuals = [np.array(cf) for cf in counterfactuals]

    N = np.array(counterfactuals).shape[-1]
    M = np.array(counterfactuals).shape[-2]

    data = np.concatenate( (data.reshape(-1,M*N),np.array(counterfactuals).reshape(-1,M* N)))
    nbrs = NearestNeighbors(n_neighbors=y).fit(np.array(data))

 
    calc=[]
    for i, row in enumerate(counterfactuals):
        ##print(row)
        row=row.reshape(-1,M* N)
        knn = nbrs.kneighbors(row, y, return_distance=False)[0]
        cf_label = labels[i] 

        for idx in knn:
            neighbour = data[idx] 
            neighbour = neighbour.reshape((1,M, N))

            individual = np.array(neighbour.tolist(), dtype=np.float64)
            input_ = torch.from_numpy(individual).reshape((1,M, N)).float()

            output = torch.nn.functional.softmax(mlmodel(input_)).detach().numpy()
            neighbour_label = np.argmax(output)
            #print('CF_Label', cf_label)
            #print('Neighbor_Label', neighbour_label)
            if not np.argmax(cf_label) == neighbour_label:
                number_of_diff_labels += 1
        calc.append([1 - (1 / ( y)) * number_of_diff_labels])

    return np.array(calc)

def yNN_timeseries(
    counterfactuals,
    mlmodel,data,
    y,labels=None
) :
    """

    """
    number_of_diff_labels = 0
    if labels is None:
        labels = [np.argmax(cf.output) for cf in counterfactuals]

    counterfactuals = [np.array(cf) for cf in counterfactuals]

    N = np.array(counterfactuals).shape[-1]
    M= np.array(counterfactuals).shape[-2]

    data = np.concatenate( (data.reshape(-1, M,N),np.array(counterfactuals).reshape(-1, M,N)))

    data=data.reshape(-1, N,M)

    nbrs = KNeighborsTimeSeries(n_neighbors=y, metric = 'dtw')
    nbrs.fit(np.array(data))
    
    calc=[]
    p=len(counterfactuals)
    for i, row in enumerate(counterfactuals):

        knn = nbrs.kneighbors(np.array(row).reshape(1,N,M), return_distance=False)

        
        cf_label = labels[i] 

        for idx in knn:
            neighbour = data[idx]
            neighbour = neighbour.reshape((1, -1))
            individual = np.array(neighbour.tolist(), dtype=np.float64)
            input_ = torch.from_numpy(individual).float().reshape(-1,M,N)
            output = torch.nn.functional.softmax(mlmodel(input_)).detach().numpy()
            neighbour_label = np.argmax(output)
            if not np.argmax(cf_label) == neighbour_label:
                number_of_diff_labels += 1
    
        calc.append(1 - (1 /(1* y)) * number_of_diff_labels)
    #if p==1: 
    return  1 - (1 / (N * y)) * number_of_diff_labels, calc


def compute_redundancy(
    fact: np.ndarray, cf: np.ndarray, mlmodel, label_value: int,mode:str
) -> int:
    red = 0
    
    if len(fact.shape) ==1:
        shape=(1,1,fact.shape[0])
    else:
        shape= fact.shape

    fact=fact.reshape(-1)
    cf=cf.reshape(-1)
    for col_idx in range(cf.shape[0]):  # input array has one-dimensional shape

        if fact[col_idx] != cf[col_idx]:
            temp_cf = np.copy(cf)

            temp_cf[col_idx] = fact[col_idx]

            individual = np.array(temp_cf.tolist(), dtype=np.float64)
            #print('Individual', individual.reshape(1,shape[-2],shape[-1]).shape)
            #if mode == 'feat':
            #     input_ = torch.from_numpy(individual.reshape(1,shape[-1],shape[-2])).float()
            #else:
            
            input_ = torch.from_numpy(individual.reshape(1,shape[-2],shape[-1])).float()

            output = torch.nn.functional.softmax(mlmodel(input_)).detach().numpy()


            temp_pred = np.argmax(output)
            #print('temp_pred',temp_pred)
            #print('label_value',label_value)

            if temp_pred == np.argmax(label_value):
                red += 1

    return red


def redundancy(original, counterfactuals, mlmodel, labels= None,mode='time') :
    """

    """
    
    if labels == None:
        labels = [np.argmax(cf.output) for cf in counterfactuals]
    df_cfs = np.array(counterfactuals)
    redun=[]
    for i in range (0,len(df_cfs)):
        redun.append(compute_redundancy(original,np.array(df_cfs[i]),mlmodel,labels[i],mode))
    return redun

def get_counterfactual_metrics(original, counterfactual,mlmodel=None,labels=None,data=None,y=None,mode='time'):
    '''
    Summarizes the CF Metrics.

    Attributes:
        original np.array: original / factual data in format (#,feat, time) or (#, time, feat)
        counterfactual np.array: counterfactualdata in format (#,feat, time) or (#, time, feat)
        ml_model Callable: Classification Model. If None, only distance metrics are calculated 
        labels np.array:
        data Tuple: 
        y: 
        mode: 
    Returns
        pd.DataFrame 
    '''
    #if mode =='feat':
    #    original=original.reshape(-1, original.shape[-1],original.shape[-2])
    #    counterfactuals=counterfactuals.reshape(-1, counterfactuals.shape[-1],counterfactuals.shape[-2])
    # TODO GET ONLY VALID DATA --> index  --> original --> CF
    cf_shape=counterfactual.shape
    #print('Input Org',original.shape)
    #print('CounterfactualOrg',counterfactual.shape)

    counterfactual=counterfactual.reshape(cf_shape[0],-1)
    original=original.reshape(cf_shape[0],-1)
    tmp = np.where(np.isin(counterfactual, [None]), np.nan,counterfactual).astype(float)
    counterfactual_valid = tmp[~np.isnan(tmp).all(axis=1)].reshape(-1,cf_shape[-2],cf_shape[-1])
    original_valid=original[~np.isnan(tmp).all(axis=1)].reshape(-1,cf_shape[-2],cf_shape[-1])
    #counterfactual_valid=[]
    #original_valid=[]
    #print('Original Valid',original_valid.shape)
    #print('Counterfactual valid',counterfactual_valid.shape)
    d1= distance_1(original_valid,counterfactual_valid)
    d2=distance_2(original_valid,counterfactual_valid)
    d3=distance_3(original_valid,counterfactual_valid)
    d4=distance_4(original_valid,counterfactual_valid)
    df = pd.DataFrame([])
    df['d1']=d1
    df['d2']=d2
    df['d3']=d3
    df['d4']=d4
    #TODO
    df['validty']=np.repeat(len(counterfactual_valid)/len(counterfactual), len(d1))
    if data is None or mlmodel is None:
        return df
    X,Y=data
 
    ynn_time_full,ynn_time_single=yNN_timeseries(counterfactual_valid, mlmodel,X,5,labels)
    red=redundancy(original_valid,counterfactual_valid, mlmodel,labels,mode)
    ynn_time_rep=np.repeat(ynn_time_full,len(red))
    df['ynn']=ynn_time_single
    df['ynn_full']=ynn_time_full
    df['red']=red
    print(df) 
    return df
