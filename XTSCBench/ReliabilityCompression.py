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
from XTSCBench.Helper import  counterfactual_manipulator
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import plotly.graph_objects as go
import numpy as np

class Compression ():
    def __init__(self,  mlmodel,explainer,metrics=None,data=None):
        """
        #TODO TCN Not working --> needs(1,1,50,50 as Input)
        Attributes: 
            typ (array): array of string, type of data, For full Iteration use ''
            datagenertaion (array): array of datageneration processes
            classificator (array): strings of classificators to be tested ['CNN','LSTM','LSTM_ATT']
            data_dir (str): data dictonary of functions
        """# 'SmallMiddle'
        super().__init__()
        self.models=mlmodel
        self.explainers=explainer
        #TODO 

        self.metrics=metrics

        #TODO IS THIS removable ? 
        self.data=data
        self.columns=[]

    def evaluate():
        raise NotImplementedError("For the Synthetic Data Based Ground Truth Evaluation, only evaluate_synthtic is implemented!.")


    def evaluate_synthetic(self,types, classificator, data_dir, num_items=100,save=None,elementwise=None, explanation_path=None,save_exp=False):
        '''
        Evaluates Roboustness on Sythetic Data.
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
        for name in reversed(list(data_full.keys())):#range(len(self.datagenerationtypes)):
                
                splitting = name.split('_')
                typ=splitting[-6]
                generation=splitting[-5]
                d_train=data_train[name]
                l_train=label_train[name]
                data=data_full[name]
                label=label_full[name]
                data_shape_1 = data.shape[1]
                data_shape_2 = data.shape[2]

                raw_data=np.copy(data)
                data, test_loaderRNN, scaler = scaling(data, label, shape_1, shape_2)
                d_train, train_loaderRNN, scaler = scaling(d_train, l_train, shape_1, shape_2)

                modelName =  name.replace('Testing','Training')
                
                for m in self.classification_models:
                    for explainer in self.explainers:
                        #print(f'Name : {name}, model: {m}, explainer: {explainer}, generation: {generation}, typ: {typ}, modelname: {modelName}')
                        s= str(type(explainer)).split('.')[-1].replace('>','')
                        sav=str(parameters_to_pandas(explainer).values)

                        if os.path.isfile(f'./Results/Explanation/{name}_{m}_{s}_{sav}.csv'):
                            number=number+1
                            continue
                       
                       
                        if does_entry_already_exist(old_data, m, generation, typ, modelName):
                            number=number+1
                            continue    
                        #{modelName}_BEST.pkl'
                        mod= torch.load(f'./XTSCBench/ClassificationModels/models/{m}/{modelName}',map_location='cpu')

                        mname=name.replace('Testing','Training')
                        print(f'RUN {number}/{total} data {name}, model {m}, salienymethod {str(type(explainer))}, params {parameters_to_pandas(explainer)}')
                        l_train=label_train[name]
                        #print(l_train)
                        d_train=data_train[name]
                        #print('Shape of Training Data',d_train.shape)
                        explainer_old=explainer
                        explainer = manipulate_exp_method(d_train, l_train, data_shape_1, data_shape_2, scaler, explainer, mod)
                        if type(explainer)== str: 
                            number +=1
                            continue
                        if explanation_path is None:
                            sal=get_explanation(data, label, data_shape_1, data_shape_2, explainer, mod)
                        else: 
                            s= str(type(explainer)).split('.')[-1].replace('>','')
                            res=np.load(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(explainer_old).values)}.csv',allow_pickle=True)
                            if type(res)== str: 
                                number=number+1
                                continue
                            if 'CNN' in str(type(mod)):
                                sal=res.reshape(-1,data_shape_2,data_shape_1)
                            else:
                                sal=res.reshape(-1,data_shape_1,data_shape_2)

                            save_expl=False
                        if save_expl:
                            s= str(type(explainer)).split('.')[-1].replace('>','')#TODO Used to be '\n'
                            with open(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(explainer).values)}.csv', 'wb') as f:
                                try:
                                    np.save(f,np.array(sal))
                                except: 
                                    print(sal)
                                    print(len(sal))
                                    
                        if 'CF' in str(type(explainer)):
                            res,data,meta,label=counterfactual_manipulator(res,data, meta=meta, data_shape_1=data_shape_1,data_shape_2=data_shape_2,scaler=scaler, raw_data=None, scaling=True, labels=label,cf_man=True)
                        if None in res:
                            res,data,meta,label=counterfactual_manipulator(res,data, meta=meta, data_shape_1=data_shape_1,data_shape_2=data_shape_2,scaler=scaler, raw_data=None, scaling=True, labels=label,cf_man=False)

                

                        masks=find_masks(np.array(sal))
                        masked_acc=get_masked_accuracy(mod,test_loaderRNN,raw_data,label,masks,scaler,generation, 'irregular',(data_shape_1,data_shape_2))

                        precision,recall=get_precision_recall(sal,raw_data,meta,masks,(data_shape_1,data_shape_2))

                        #Get AUC, AUR, AUP and AUPR
                        AUP, AUR, AUPR, AUC= get_accuracy_metrics(precision, recall, masked_acc)
                        if elementwise is not None:
                            if not os.path.isdir(f'{elementwise}/degenerate'):
                                os.mkdir(f'{elementwise}/degenerate')
                            pd.DataFrame(zip(recall,precision,masked_acc),columns=['Recall','Precision','ACC']).to_csv(f'{elementwise}/degenerate/{name}_{m}_{str(parameters_to_pandas(explainer).values)}.csv')
                        
                        
                        row_summary=pd.DataFrame([[modelName,typ,generation,m,AUP,AUR,AUPR,AUC]],columns=SummaryTableCol)
                        new_row_summary=pd.concat([row_summary,parameters_to_pandas(explainer)], axis = 1)


                        SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True)
                        if save is None:
                            SummaryTable.to_csv('temp.csv')
                        else: 
                            SummaryTable.to_csv(f'{save}')

                        number=number+1

        return SummaryTable

   
    
    def compare_dataset(self,from_file=None):
        '''
        Replication of Table 1 

        '''
        data = pd.read_csv(from_file).drop(columns='Unnamed: 0')
       
        '''Combine method with parameters'''
        method_full = data['method']
        for a in data.columns: 
            if a not in self.SummaryTableCol:
                method_full += '_' + data[a].astype(str)
        data['method_full']=method_full
        #TODO First Sort Dataset than Results
        return pd.pivot_table(data, index="method_full",columns=["Typ"],values=["AUPR","AUP","AUR","AUC"],aggfunc='mean')#.groupby(level='Typ',axis=1))
   

