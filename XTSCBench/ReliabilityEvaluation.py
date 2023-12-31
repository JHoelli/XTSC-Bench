import pickle
from XTSCBench.Evaluation import Evaluation
import pandas as pd 
import numpy as np
import torch
from XTSCBench.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from XTSCBench.metrics.reliability_metrics import get_reliability_metrics
from sklearn import preprocessing as pre
import os 
from XTSCBench.metrics.synthetic_helper import load_synthetic_data,manipulate_exp_method,scaling, get_explanation,does_entry_already_exist
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import plotly.graph_objects as go
import numpy as np
from XTSCBench.Helper import  counterfactual_manipulator

class ReliabilityEvaluation (Evaluation):
    
    def __init__(self, mlmodel,explainer,metrics=None):
        #super().__init__(mlmodel)
        self.models=mlmodel
        self.explainers=explainer
        self.metrics=metrics

    def evaluate(self, items,label,model,meta, exp=None, mode='time', aggregate=False):
        '''
        Enables evaluation on custom model, expalainer and data.
        Attributes:
            items np.array: items to be used in the evaluation
            label np.array: Labels of items 
            model torch.nn.Module:
            exp np.array: Defaults to None, If Explanation is already calculated
            mode str: first dimension 'time' or 'feat'
            aggregate bool: Return mean and std 
        Returns: 
            pd.DataFrame
        
        '''

        row_summary=None
        SummaryTable = pd.DataFrame([])
        data_shape_1= items.shape[-2]
        data_shape_2= items.shape[-1]
        
        for baseline in self.explainers:      
            exp_new=get_explanation(items, label, data_shape_1, data_shape_2, baseline, model,mode)
            exp_new=np.array(exp_new).reshape(-1, data_shape_1,data_shape_2)
            row_summary=get_reliability_metrics(items, exp_new,model,label,meta,(data_shape_1,data_shape_2),synthtic=False,additional_metrics=self.metrics)
            if not aggregate:
                df=parameters_to_pandas(baseline)
                newdf = pd.DataFrame(np.repeat(df.values, len(row_summary), axis=0))
                newdf.columns = df.columns
                newdf['explanation'] =np.repeat(str(type(baseline)).split('.')[-1], len(newdf), axis=0)
                new_row_summary = pd.concat([row_summary,newdf], axis = 1)
            if aggregate:
                means = row_summary.mean().add_suffix('_mean')
                std = row_summary.std().add_suffix('_std')
                new= pd.concat([means,std]).to_frame().T
                new_row_summary=pd.concat([new,parameters_to_pandas(baseline)], axis = 1)
 
            if len(SummaryTable)== 0:
                SummaryTable=new_row_summary
            else:
                SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True)

        if exp is not None: 
            row_summary= get_reliability_metrics(items, exp,model,label,meta,(data_shape_1,data_shape_2),synthtic=False, additional_metrics=self.metrics)
            if not aggregate:
                newdf['explanation'] =np.repeat('custom', len(row_summary), axis=0)
                new_row_summary = pd.concat([row_summary,newdf], axis = 1)
            if aggregate:
                means = row_summary.mean().add_suffix('_mean')
                std = row_summary.std().add_suffix('_std')
                new_row_summary= pd.concat([means,std]).to_frame().T
                new_row_summary['explanation']=['custom']
            SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True,axis=1)     
        return SummaryTable

 
    def evaluate_synthetic(self,types, classificator, data_dir, num_items=100,save=None,elementwise=None, explanation_path=None, save_exp=None):
        '''
        Evaluates Reliability on Sythetic Data.
        Attributes: 
            types List: specify the information feature type, the generation process type or a full dataset name e.g., ['SmallMiddle', 'ARIMA'], if you want to run on all use ''
            classificator List: Which Pretrained CLassifier to use, currently ['LSTM', 'CNN', 'LSTMATT'] are available.
            data_dir str: Path to directory of the synthic data (in case it is downloaded to a different folder)
            num_items int: Number Items to runt the evaluation on. Default to 100. 
            save str: Path to save Results. If File already exits, current results are appended.
            explanation_path str : If Metric calculation based on previous Explanaton Calculation, provide path to explanbation here.
            elementwise str: If the results should be stored itemwise, put path to the desired Folder here.
        '''
        self.types=types
        self.classification_models= classificator
     
        '''Load Data'''
        if type(self.types[0])==str: 
            data_train, meta_train, label_train,data_full,meta_full,label_full=load_synthetic_data(self.types,data_dir,return_train=True)
        old_data=None

        '''Tries to Load Existing Data / Previous Runs'''
        if save is not None:
            if os.path.isfile(save):
                try:
                    old_data=pd.read_csv(save).drop('Unamed: 0')
                except:
                    old_data=pd.read_csv(save)

        '''Loop through possibilities Starts'''
        SummaryTableCol=['Datasets','Typ','Generation','models']

        SummaryTable=pd.DataFrame([])
        number= 0
        total = len(list(data_full.keys())) *len(self.classification_models) * len(self.explainers)

        for name in data_full.keys():#range(len(self.datagenerationtypes)):
                
                splitting = name.split('_')
                typ=splitting[-6]
                generation=splitting[-5]
                d_train=data_train[name]
                l_train=label_train[name]
                data=data_full[name]
                label=label_full[name]
                data_shape_1 = data.shape[1]
                data_shape_2 = data.shape[2]
                meta=meta_full[name]

                data, test_loaderRNN, scaler = scaling(data, label,  data_shape_1,  data_shape_2)
                d_train, train_loaderRNN, scaler = scaling(d_train, l_train,  data_shape_1,  data_shape_2)

                modelName =  name.replace('Testing','Training')
                
                for m in self.classification_models:
                    if 'CNN' in str(type(m)):
                        mode='feat'
                    else:
                        mode='time' 
                    for explainer in self.explainers:
                        print(f'RUN {number}/{total} data {name}, model {m}, salienymethod {str(type(explainer))}, params {parameters_to_pandas(explainer)}')
                       
                        '''Check wheather Calculation already exists'''
                        if does_entry_already_exist(old_data, m, generation, typ, modelName):
                            number =number+1
                            continue  
                        '''Load Model and Manipulate Explainer'''
                        mod= torch.load(f'./XTSCBench/ClassificationModels/models/{m}/{modelName}',map_location='cpu')
                        #old_explainer = explainer
                        explainer = manipulate_exp_method(d_train, l_train, data_shape_1, data_shape_2, scaler, explainer, mod)

                        if type(explainer) ==str: 
                            print('Predictor returns constant predictor')
                            number +=1
                            continue      
                        
                        '''Calculate Explanations'''
                        data=data_full[name][:num_items]
                        label=label_full[name][:num_items]
                        meta=meta_full[name][:num_items]
                        res=[]
                        s= str(type(explainer)).split('.')[-1].replace('>','')
                        if explanation_path is None or f'{name}_{m}_{s}_{str(parameters_to_pandas(explainer).values)}.csv' not in os.listdir(explanation_path):
                            res=get_explanation(data[:num_items], label[:num_items], data_shape_1, data_shape_2, explainer, mod,mode)
                            res=np.array(res)
                            if save_exp is not None:
                                s= str(type(explainer)).split('.')[-1].replace('>','')#TODO Used to be '\n'
                                with open(f'{explanation_path}/{name}_{m}_{s}_{str(parameters_to_pandas(explainer).values)}.csv', 'wb') as f:
                                    try:
                                        np.save(f,np.array(res))
                                    except: 
                                        print(res)
                                        print(len(res)) 
                        else:                             
                            res=np.load(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(explainer).values)}.csv',allow_pickle=True)[:num_items]
                            if type(res)== str: 
                                continue
                        if None in res: 
                            res,data,meta,label=counterfactual_manipulator(res,data, meta=meta, data_shape_1=data_shape_1,data_shape_2=data_shape_2,scaler=scaler, raw_data=None, scaling=True, labels=label,cf_man=False)

                        num_items=len(res)
                        if 'CNN' in str(type(mod)):
                            mode='feat'
                            #data = np.swapaxes(data,-1,-2)#.reshape(-1,shape_2,shape_1)
                            res=np.swapaxes(res,-1,-2)#.reshape(-1,shape_2,shape_1)
                        #else:
                        #    mode='time'
                        #    data = data.reshape(-1,data_shape_1,data_shape_2)
                        #res=res.reshape(-1,data_shape_1,data_shape_2)
                        exp=res.reshape(-1,res.shape[-2],res.shape[-1])
                        if len(exp)== 0:
                            continue
                        exp=np.array(exp)[:num_items]                  
                        distances = get_reliability_metrics(data[:num_items], exp[:num_items],mod,label[:num_items],meta[:num_items],(data_shape_1,data_shape_2), mode= mode)
 
                        number =number+1

                        '''Savings Section'''
                        if elementwise is not None:
                            df=parameters_to_pandas(explainer)
                            newdf = pd.DataFrame(np.repeat(df.values, len(distances), axis=0))
                            newdf.columns = df.columns
                            newdf['explanation'] =np.repeat(str(type(explainer)).split('.')[-1], len(newdf), axis=0)
                            distances_man = pd.concat([distances,newdf], axis = 1)
                            distances_man.to_csv(f'{elementwise}/Reliability/{name}_{m}_{str(parameters_to_pandas(explainer).values)}.csv')

                        means = distances.mean().add_suffix('_mean')
                        std = distances.std().add_suffix('_std')

                        su =pd.DataFrame([[modelName,typ,generation,m]], columns=SummaryTableCol)

                        new= pd.concat([means,std]).to_frame().T
                        
                        new_row_summary=pd.concat([new,parameters_to_pandas(explainer)], axis = 1)
                        
                        new_row_summary= pd.concat([new_row_summary, su], axis = 1)
                        
                        if len(SummaryTable)>0:
                            SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True)
                        else: 
                            SummaryTable=new_row_summary


                        if save is None:
                            SummaryTable.to_csv('temp.csv')
                        else: 
                            SummaryTable.to_csv(f'{save}')
        return SummaryTable