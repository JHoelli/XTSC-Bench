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
import torch
from XTSCBench.Evaluation import Evaluation
from XTSCBench.metrics.synthetic_helper import load_synthetic_data, manipulate_exp_method
from XTSCBench.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from XTSCBench.metrics.counterfactual_metrics import *
from XTSCBench.metrics.synthetic_helper import load_synthetic_data,manipulate_exp_method,scaling, get_explanation,does_entry_already_exist, get_preds
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn
import torch.utils.data as data_utils
import seaborn as sns
from torch.utils.data import DataLoader

class CounterfactualEvaluation(Evaluation):
    """
    Calculates the L0, L1, L2, and L-infty distance measures.
    """

    def __init__(self, mlmodel=None,explainer=None,data=None):
        #super().__init__(mlmodel)
        self.models=mlmodel
        self.explainers=explainer
        self.columns = ["d1", "d2", "d3", "d4","yNN Time", "Redundancy"]
        self.data=data
    #items,label,model, expla=None, mode='time', aggregate=False
    def evaluate(self,items,label,model, expla=None, mode='time', aggregate=False):
        '''
        If Both are None Fallback on Synthetic
        '''
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
            exp=get_explanation(items, label, data_shape_1, data_shape_2, baseline, model,mode)
            exp=np.array(exp).reshape(-1, data_shape_1,data_shape_2)
            row_summary= get_counterfactual_metrics(items,exp,model,label,data=None,y=None,mode='time')
            #get_complexity_metrics(items,exp,model,label,baseline,mode=mode, additional_metrics=self.metrics)
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

        if expla is not None: 
            row_summary= get_counterfactual_metrics(items,exp,model,label,data=None,y=None,mode='time')
            if not aggregate:
                newdf=pd.DataFrame([])
                newdf['explanation'] =np.repeat('custom', len(row_summary), axis=0)
                new_row_summary = pd.concat([row_summary,newdf], axis = 1)
            if aggregate:
                means = row_summary.mean().add_suffix('_mean')
                std = row_summary.std().add_suffix('_std')
                new_row_summary= pd.concat([means,std]).to_frame().T
                new_row_summary['explanation']= ['custom']
            if len(SummaryTable)== 0:
                SummaryTable=new_row_summary
            else:
                SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True) 
        return SummaryTable
 
    def evaluate_synthetic(self,types, classificator, data_dir, num_items=100,save=None,elementwise=None, explanation_path=None):
        '''
        Evaluates Counterfacual Approaches on Sythetic Data.

        Attributes: 
            types List: specify the information feature type, the generation process type or a full dataset name e.g., ['SmallMiddle', 'ARIMA'], if you want to run on all use ''
            classificator List: Which Pretrained CLassifier to use, currently ['LSTM', 'CNN', 'LSTMATT'] are available.
            data_dir str: Path to directory of the synthic data (in case it is downloaded to a different folder)
            num_items int: Number Items to runt the evaluation on. Default to 100. 
            save str: Path to save Results. If File already exits, current reusults are appended
            explanation_path str : If Metric calculation based on previous Explanaton Calculation, provide path to explanbation here.
        '''
        self.types=types
        self.classification_models= classificator
        
        '''Load Data'''
        if type(self.types[0])==str: 
            data_train, meta_train, label_train,data_full,_,label_full=load_synthetic_data(self.types,data_dir,return_train=True)
        old_data=None

        '''Tries to Load Existing Data / Previous Runs'''
        if save is not None:
            if os.path.isfile(save):
                try:
                    old_data=pd.read_csv(save).drop('Unamed: 0')
                except:
                    old_data=pd.read_csv(save)

        '''Loop through possibilities Starts'''
        SummaryTableCol=['Datasets','Typ','Generation','models',*self.columns]

        SummaryTable=pd.DataFrame(columns=SummaryTableCol)

        for name in data_full.keys():#range(len(self.datagenerationtypes)):
                
                splitting = name.split('_')
                typ=splitting[-6]
                generation=splitting[-5]
                d_train=data_train[name]
                l_train=label_train[name]
                data=data_full[name]
                label=label_full[name]
                shape_1 = data.shape[1]
                shape_2 = data.shape[2]

                raw_data=np.copy(data)
                data, test_loaderRNN, scaler = scaling(data, label, shape_1, shape_2)
                d_train, train_loaderRNN, scaler = scaling(d_train, l_train, shape_1, shape_2)

                modelName =  name.replace('Testing','Training')
                
                for m in self.classification_models:
                    for explainer in self.explainers:

                        '''Check wheather Calculation already exists'''
                        if does_entry_already_exist(old_data, m, generation, typ, modelName):
                            continue  
                        '''Load Model and Manipulate Explainer'''
                        mod= torch.load(f'./XTSC-Bench/ClassificationModels/models_new/{m}/{modelName}',map_location='cpu')
                        mname=name.replace('Testing','Training')
                        explainer = manipulate_exp_method(d_train, l_train, shape_1, shape_2, scaler, explainer, mod)

                        if type(explainer) ==str: 
                            #TODO add Log?
                            continue
                        
                        '''Calculate Explanations'''
                        y_pred=[]
                        res=[]
                        if explanation_path is None: 
                            
                            for x,y in zip(data[:num_items],label[:num_items]):
                                if 'CNN' in str(type(mod)):
                                    x =torch.from_numpy(np.swapaxes(x,-2,-1)).float().reshape(1,shape_2,shape_1)
                                else: 
                                    x =torch.from_numpy(x).float().reshape(1,shape_1,shape_2)
                                pred= mod(x).detach().numpy()
                                # Call Explainer
                                counterfactual,label_cf=explainer.explain(x.detach().numpy(),np.array([np.argmax(pred[0])]))
                                #print('CF',counterfactual)
                                y_pred.append(np.argmax(label_cf))
                                res.append(counterfactual)
                        else:                             
                            s= str(type(explainer)).split('.')[-1].replace('>','')
                            res=np.load(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(explainer).values)}.csv')
                            y_pred = get_preds(mod, res)
               
            
                        distances = get_counterfactual_metrics(np.array(data[:num_items]), np.array(res[:num_items]),mlmodel=mod,labels=y_pred,data=(d_train,l_train),mode=explainer.mode)
                    

                        '''Savings Section'''
                        #row_summary=pd.DataFrame(distances,columns=self.columns)
                        if elementwise is not None:
                            df=parameters_to_pandas(explainer)
                            newdf = pd.DataFrame(np.repeat(df.values, len(distances), axis=0))
                            newdf.columns = df.columns
                            newdf['explanation'] =np.repeat(str(type(explainer)).split('.')[-1], len(newdf), axis=0)
                            distances_man = pd.concat([distances,newdf], axis = 1)
                            distances_man.to_csv(f'{elementwise}/{name}_{m}_{str(parameters_to_pandas(explainer).values)}_Counterfactual.csv')

                        means = distances.mean().add_suffix('_mean')
                        std = distances.std().add_suffix('_std')
                        new= pd.concat([means,std]).to_frame().T
                        new_row_summary=pd.concat([new,parameters_to_pandas(explainer)], axis = 1)
                        SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True)

                        if save is None:
                            SummaryTable.to_csv('temp.csv')
                        else: 
                            SummaryTable.to_csv(f'{save}')
                        #print(row_summary)
