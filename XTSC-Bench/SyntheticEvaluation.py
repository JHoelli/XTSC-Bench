import pickle
import pandas as pd 
import numpy as np
import torch
#from Benchmarking.Models.LSTM import LSTM
#from Benchmarking.Models.TCN import TCN
from Benchmarking.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from Benchmarking.metrics.synthetic_metrics import find_masks,get_masked_accuracy, get_precision_recall,get_accuracy_metrics, get_precision_recall_plain, get_quantus_metrics
import os 
from Benchmarking.metrics.synthetic_helper import load_synthetic_data,manipulate_exp_method,scaling, get_explanation,does_entry_already_exist
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import plotly.graph_objects as go
import numpy as np

class Synthetic ():
    def __init__(self,saliency_functions,typ=['_'],  classificator=['CNN','LSTM'], data_dir='./Benchmarking/data/multivariate/',tolerance = 0):
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
        self.SummaryTableCol=['Datasets','Typ','Generation','models','AUP','AUR','AUPR','AUC','method']
        self.tolerance=tolerance

    def evaluate():
        raise NotImplementedError("For the Synthetic Data Based Ground Truth Evaluation, only evaluate_synthtic is implemented!.")


    def evaluate_synthetic(self,num_items=100,save=None, elementwise= None, degenerate=False, save_expl=True,quantus=False,explanation_path=None):
        '''
        saliency (func): function that takes item and labels as input and returns the interpretation (somentimes lambda function necessary)
        #TODO Is devision between test and Train clear ? 
        '''
        number=0
        
        start_loading=time.time()
        saliency_functions=self.saliency_functions
        if type(self.types[0])==str: 
            #FILTERS DATA 
            data_train, meta_train, label_train, data_full, meta_full, label_full=load_synthetic_data(self.types,self.data_dir,True)
        old_data=None
        if os.path.isfile(save):
            try:
                old_data=pd.read_csv(save).drop('Unamed: 0')
            except:
                old_data=pd.read_csv(save)
        '''Loop through possibilities Starts'''
        SummaryTableCol=['Datasets','Typ','Generation','models','AUP','AUR','AUPR','AUC','Plain_Precision','Plain_Recall','Plain_Acc']
        if old_data is None:
            SummaryTable=pd.DataFrame(columns=SummaryTableCol)
        else: 
            SummaryTable = old_data

        total = len(list(data_full.keys())) *len(self.classification_models) * len(self.saliency_functions)


        for name in data_full.keys():#range(len(self.datagenerationtypes)):
                
                start_data=time.time()
                splitting = name.split('_')
                typ=splitting[-6]
                generation=splitting[-5]
                data=data_full[name][:num_items]
                label=label_full[name][:num_items]
                meta=meta_full[name][:num_items]
                data_shape_1=data.shape[1]
                data_shape_2=data.shape[2]

                
                raw_data=np.copy(data)

                data, test_loaderRNN, scaler = scaling(data, label, data_shape_1, data_shape_2)
                modelName =  name.replace('Testing','Training')
               
                for m in self.classification_models:
                    #print(f'MODEL {m}')
                    start_class=time.time()
                    for saliency in saliency_functions:
                        #print(f'Name : {name}, model: {m}, saliency: {saliency}, generation: {generation}, typ: {typ}, modelname: {modelName}')
                        s= str(type(saliency)).split('.')[-1].replace('>','')
                        sav=str(parameters_to_pandas(saliency).values)

                        if os.path.isfile(f'./Results/Explanation/{name}_{m}_{s}_{sav}.csv'):
                            number=number+1
                            continue
                       
                       
                        if does_entry_already_exist(old_data, m, generation, typ, modelName):
                            number=number+1
                            continue    
                        #{modelName}_BEST.pkl'
                        mod= torch.load(f'./Benchmarking/ClassificationModels/models_new/{m}/{modelName}',map_location='cpu')

                        mname=name.replace('Testing','Training')
                        print(f'RUN {number}/{total} data {name}, model {m}, salienymethod {str(type(saliency))}, params {parameters_to_pandas(saliency)}')
                        l_train=label_train[name]
                        #print(l_train)
                        d_train=data_train[name]
                        #print('Shape of Training Data',d_train.shape)
                        saliency_old=saliency
                        saliency = manipulate_exp_method(d_train, l_train, data_shape_1, data_shape_2, scaler, saliency, mod)
                        if type(saliency)== str: 
                            #with open(f'{elementwise}/plain/{name}_{m}_{str(parameters_to_pandas(saliency_old).values)}.txt', 'w') as f:
                            #    f.write(f'{saliency}')
                            with open(f'{elementwise}/Debug.txt', 'w') as f:
                                f.write(f'{name}_{m}_{str(parameters_to_pandas(saliency_old).values)} -- Constant Predictore')
                                f.close()
                            if save_expl:
                                s= str(type(saliency_old)).split('.')[-1].replace('>','')#
                                sav=str(parameters_to_pandas(saliency_old).values)
                                with open(f'./Results/Explanation/{name}_{m}_{s}_{sav}.csv', 'wb') as f:
                                    np.save(f,np.array([]))
                                    f.close()
                            continue
                        if explanation_path is None:
                            try: 
                                sal=get_explanation(data, label, data_shape_1, data_shape_2, saliency, mod)
                            except: 
                                continue
                        else: 
                            s= str(type(saliency)).split('.')[-1].replace('>','')
                            res=np.load(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(saliency).values)}.csv')
                            if 'CNN' in str(type(mod)):
                                sal=res.reshape(-1,data_shape_2,data_shape_1)
                            else:
                                sal=res.reshape(-1,data_shape_1,data_shape_2)

                            save_expl=False
                        if save_expl:
                            s= str(type(saliency)).split('.')[-1].replace('>','')#TODO Used to be '\n'
                            with open(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(saliency).values)}.csv', 'wb') as f:
                                try:
                                    np.save(f,np.array(sal))
                                except: 
                                    print(sal)
                                    print(len(sal))
                                    
                       
                        if 'CF' in str(type(saliency)):

                            sal=np.array(sal).reshape(-1,data_shape_1*data_shape_2)
                            # FOR THIS CALC DROP All None
                            tmp = np.where(np.isin(sal, [None]), np.nan,sal).astype(float)
                            new_arr = tmp[~np.isnan(tmp).all(axis=1)]

                            if len(new_arr) <1:
                                print('All values are None')
                                continue
                            else: 
                                sal = new_arr

                            raw_data_scaled = scaler.transform(raw_data.reshape(-1,data_shape_1*data_shape_2))
                            # Index
                            inde= np.where(sal.reshape(-1,data_shape_1*data_shape_2)!=raw_data_scaled.reshape(-1,data_shape_1*data_shape_2))
                            sal_man= np.zeros_like(sal)

                            sal_man[inde]=1
                            sal=sal_man.reshape(-1,data_shape_1,data_shape_2)

                    
                        if degenerate:

                            masks=find_masks(np.array(sal))
                            masked_acc=get_masked_accuracy(mod,test_loaderRNN,raw_data,label,masks,scaler,generation, 'irregular',(data_shape_1,data_shape_2))

                            precision,recall=get_precision_recall(sal,raw_data,meta,masks,(data_shape_1,data_shape_2))
                       
                            start=time.time()
                            #Get AUC, AUR, AUP and AUPR
                            AUP, AUR, AUPR, AUC= get_accuracy_metrics(precision, recall, masked_acc)
                            if elementwise is not None:
                                if not os.path.isdir(f'{elementwise}/degenerate'):
                                    os.mkdir(f'{elementwise}/degenerate')
                                pd.DataFrame(zip(recall,precision,masked_acc),columns=['Recall','Precision','ACC']).to_csv(f'{elementwise}/degenerate/{name}_{m}_{str(parameters_to_pandas(saliency).values)}.csv')
                        
                        
                            row_summary=pd.DataFrame([[modelName,typ,generation,m,AUP,AUR,AUPR,AUC,plain_precision,plain_recall,plain_acc]],columns=SummaryTableCol)
                            new_row_summary=pd.concat([row_summary,parameters_to_pandas(saliency)], axis = 1)

                        else: 
                            row_summary=pd.DataFrame([[modelName,typ,generation,m,'NA','NA','NA','NA',plain_precision,plain_recall,plain_acc]],columns=SummaryTableCol)
                            new_row_summary=pd.concat([row_summary,parameters_to_pandas(saliency)], axis = 1)

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
   

