import pickle
import pandas as pd 
import numpy as np
import torch
#from XTSCBench.Models.LSTM import LSTM
#from XTSCBench.Models.TCN import TCN
from XTSCBench.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from XTSCBench.metrics.synthetic_metrics import find_masks,get_masked_accuracy, get_precision_recall,get_accuracy_metrics, get_precision_recall_plain, get_quantus_metrics
import os 
from XTSCBench.metrics.synthetic_helper import load_synthetic_data,manipulate_exp_method,scaling, get_explanation,does_entry_already_exist
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import plotly.graph_objects as go
import numpy as np


class RunExp ():
    def __init__(self,saliency_functions,typ=['_'],  classificator=['CNN','LSTM'], data_dir='./XTSCBench/data/multivariate/',tolerance = 0):
        """
        #TODO TCN Not working --> needs(1,1,50,50 as Input)
        Attributes: 
            typ (array): array of string, type of data, For full Iteration use ''
            datagenertaion (array): array of datageneration processes
            classificator (array): strings of classificators to be tested ['CNN','LSTM','LSTM_ATT']
            data_dir (str): data dictonary of functions
        """# 'SmallMiddle'
        super().__init__()
        self.types=typ
        self.classification_models=classificator
        self.saliency_functions=saliency_functions
        self.data_dir=data_dir


    def calculate_explanations(self,num_items=100, save_expl=True,explanation_path=None):
        '''
        saliency (func): function that takes item and labels as input and returns the interpretation (somentimes lambda function necessary)
        #TODO Is devision between test and Train clear ? 
        '''
        number=0
        
        saliency_functions=self.saliency_functions
        if type(self.types[0])==str: 
            data_train, meta_train, label_train, data_full, meta_full, label_full=load_synthetic_data(self.types,self.data_dir,True)

        total = len(list(data_full.keys())) *len(self.classification_models) * len(self.saliency_functions)

        for name in data_full.keys():#range(len(self.datagenerationtypes)):
                data=data_full[name][:num_items]
                label=label_full[name][:num_items]
                data_shape_1=data.shape[1]
                data_shape_2=data.shape[2]

                data, test_loaderRNN, scaler = scaling(data, label, data_shape_1, data_shape_2)
                modelName =  name.replace('Testing','Training')
               
                for m in self.classification_models:
                    for saliency in saliency_functions:
                        s= str(type(saliency)).split('.')[-1].replace('>','')
                        sav=str(parameters_to_pandas(saliency).values)
                        if os.path.isfile(f'./Results/Explanation/{name}_{m}_{s}_{sav}.csv'):
                            number=number+1
                            continue
                       
                        mod= torch.load(f'./XTSCBench/ClassificationModels/models/{m}/{modelName}',map_location='cpu')

                        mname=name.replace('Testing','Training')
                        print(f'RUN {number}/{total} data {name}, model {m}, salienymethod {str(type(saliency))}, params {parameters_to_pandas(saliency)}')
                        l_train=label_train[name]
                        d_train=data_train[name]
                        saliency_old=saliency
                        saliency = manipulate_exp_method(d_train, l_train, data_shape_1, data_shape_2, scaler, saliency, mod)
                        if type(saliency)== str:
                            # In case of constant Predictor add empty explanation 
                            number=number+1
                            if save_expl:
                                s= str(type(saliency_old)).split('.')[-1].replace('>','')#
                                sav=str(parameters_to_pandas(saliency_old).values)
                                with open(f'./Results/Explanation/{name}_{m}_{s}_{sav}.csv', 'wb') as f:
                                    np.save(f,np.array([]))
                                    f.close()
                            continue
                      
                        if explanation_path is None:
                            sal=get_explanation(data, label, data_shape_1, data_shape_2, saliency, mod)

                        if save_expl:
                            s= str(type(saliency)).split('.')[-1].replace('>','')#TODO Used to be '\n'
                            with open(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(saliency).values)}.csv', 'wb') as f:
                                np.save(f,np.array(sal))
                        number=number+1
                                    
