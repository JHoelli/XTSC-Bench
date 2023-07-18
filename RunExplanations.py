import pandas as pd 
import numpy as np
import torch
from XTSCBench.metrics.metrics_helper import parameters_to_pandas
import os 
from XTSCBench.metrics.synthetic_helper import load_synthetic_data,manipulate_exp_method,scaling, get_explanation


class RunExp ():
    def __init__(self,explainer_functions,typ=['_'],  classificator=['CNN','LSTM'], data_dir='./XTSCBench/data/multivariate/',tolerance = 0):
        '''
        Attributes:
            explainer_function List: List of explainer instantiation.
            typ List: Type of Data used for evaluation, all data defaults to ['_']
            classificatior List: Type of Classificator to evaluate ['CNN','LSTM']
            data_dir List: path to uni or multivariate data
        '''
        super().__init__()
        self.types=typ
        self.classification_models=classificator
        self.explainer_functions=explainer_functions
        self.data_dir=data_dir


    def calculate_explanations(self,num_items=100):
        '''
        Helper Function to Calculate explanations for the synthetic data and trained models. Explanations are saved in ./Results/Explanation. 
        Attributes: 
            num_items int: Number of items to calculate explanations for.
       
        '''
        number=0
        
        explainer_functions=self.explainer_functions
        if type(self.types[0])==str: 
            data_train, meta_train, label_train, data_full, meta_full, label_full=load_synthetic_data(self.types,self.data_dir,True)

        total = len(list(data_full.keys())) *len(self.classification_models) * len(self.explainer_functions)

        for name in data_full.keys():#range(len(self.datagenerationtypes)):
                data=data_full[name][:num_items]
                label=label_full[name][:num_items]
                data_shape_1=data.shape[1]
                data_shape_2=data.shape[2]

                data, test_loaderRNN, scaler = scaling(data, label, data_shape_1, data_shape_2)
                modelName =  name.replace('Testing','Training')
               
                for m in self.classification_models:
                    for explainer in explainer_functions:
                        s= str(type(explainer)).split('.')[-1].replace('>','')
                        sav=str(parameters_to_pandas(explainer).values)
                        if os.path.isfile(f'./Results/Explanation/{name}_{m}_{s}_{sav}.csv'):
                            number=number+1
                            continue
                       
                        mod= torch.load(f'./XTSCBench/ClassificationModels/models/{m}/{modelName}',map_location='cpu')

                        mname=name.replace('Testing','Training')
                        print(f'RUN {number}/{total} data {name}, model {m}, salienymethod {str(type(explainer))}, params {parameters_to_pandas(explainer)}')
                        l_train=label_train[name]
                        d_train=data_train[name]
                        explainer_old=explainer
                        explainer = manipulate_exp_method(d_train, l_train, data_shape_1, data_shape_2, scaler, explainer, mod)
                        if type(explainer)== str:
                            # In case of constant Predictor add empty explanation 
                            number=number+1
                    
                            s= str(type(explainer_old)).split('.')[-1].replace('>','')#
                            sav=str(parameters_to_pandas(explainer_old).values)
                            with open(f'./Results/Explanation/{name}_{m}_{s}_{sav}.csv', 'wb') as f:
                                np.save(f,np.array([]))
                                f.close()
                            continue
                      
                        
                        sal=get_explanation(data, label, data_shape_1, data_shape_2, explainer, mod)

                      
                        s= str(type(explainer)).split('.')[-1].replace('>','')#TODO Used to be '\n'
                        with open(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(explainer).values)}.csv', 'wb') as f:
                            np.save(f,np.array(sal))
                        number=number+1
                                    
