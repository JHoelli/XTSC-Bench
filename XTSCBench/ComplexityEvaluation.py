from XTSCBench.Evaluation import Evaluation
import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import numpy as np
from XTSCBench.metrics.synthetic_helper import get_preds,load_synthetic_data,manipulate_exp_method,scaling, get_explanation,does_entry_already_exist
import os
from XTSCBench.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from XTSCBench.metrics.complexity_metrics import get_complexity_metrics
from XTSCBench.metrics.synthetic_helper import get_explanation
import quantus

class ComplexityEvaluation(Evaluation):

    def __init__(self,  mlmodel=None,explainer=None,metrics=None):
        '''
        Attributes: 
            mlmodel: Model to be evaluated
            explainer list: List of explainer instances to evaluate 
            metrics list: List of Quantus Complexity Metrics to be added to the defaul setting
        '''
        super().__init__(mlmodel)
        self.models=mlmodel
        self.explainers=explainer
        self.metrics=metrics


    def evaluate(self, items,label,model, expla=None, mode='time', aggregate=False):
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
            exp=get_explanation(items, label, data_shape_1, data_shape_2, baseline, model)
            exp=np.array(exp).reshape(-1, data_shape_1,data_shape_2)
            row_summary=get_complexity_metrics(items,exp,model,label,baseline,mode=mode)
            #print(row_summary)
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
                #print(new)
                new_row_summary=pd.concat([new,parameters_to_pandas(baseline)], axis = 1)
                #print(new_row_summary)
                
            if len(SummaryTable)== 0:
                SummaryTable=new_row_summary
            else:
                SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True)

        if expla is not None: 
            row_summary=get_complexity_metrics(items,expla,model,label,baseline,mode=mode)
            if not aggregate:
                #df=parameters_to_pandas(baseline)
                #newdf = pd.DataFrame(np.repeat(df.values, len(row_summary), axis=0))
                #newdf.columns = df.columns
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
        Evaluates Complexity on Sythetic Data.
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
                data=data_full[name][:num_items]
                label=label_full[name][:num_items]
                shape_1 = data.shape[1]
                shape_2 = data.shape[2]

                raw_data=np.copy(data)
                data, test_loaderRNN, scaler = scaling(data, label, shape_1, shape_2)
                d_train, train_loaderRNN, scaler = scaling(d_train, l_train, shape_1, shape_2)

                modelName =  name.replace('Testing','Training')
                
                for m in self.classification_models:
                    for explainer in self.explainers:
                        print(f'RUN {number}/{total} data {name}, model {m}, salienymethod {str(type(explainer))}, params {parameters_to_pandas(explainer)}')

                        '''Check wheather Calculation already exists'''
                        if does_entry_already_exist(old_data, m, generation, typ, modelName):
                            number =number+1
                            continue  
                        '''Load Model and Manipulate Explainer'''
                        mod= torch.load(f'./XTSC-Bench/ClassificationModels/models_new/{m}/{modelName}',map_location='cpu')
                        explainer = manipulate_exp_method(d_train, l_train, shape_1, shape_2, scaler, explainer, mod)

                        if type(explainer) ==str: 
                            #TODO add Log?
                            print('Predictor returns constant predictoe')
                            continue
                        
                        '''Calculate Explanations'''
                        res=[]
                        if explanation_path is None:
                            print('THIS IS NOT SUPPOSE TO HAPPEN')
                            res=get_explanation(data, label, shape_1, shape_2, explainer, mod)
                        else:                             
                            s= str(type(explainer)).split('.')[-1].replace('>','')
                            # TODO Make this work for native guide without expect
                            try: 
                                res=np.load(f'./Results/Explanation/{name}_{m}_{s}_{str(parameters_to_pandas(explainer).values)}.csv')[:num_items]
                            except: 
                                continue
                            if 'CNN' in str(type(mod)):
                                res=res.reshape(-1,shape_2,shape_1)
                            else:
                                res=res.reshape(-1,shape_1,shape_2)
                        if 'CNN' in str(type(mod)):
                            mode='feat'
                            data = data.reshape(-1,shape_2,shape_1)
                            res=res.reshape(-1,shape_2,shape_1)
                        else:
                            mode='time'
                            data = data.reshape(-1,shape_1,shape_2)
                            res=res.reshape(-1,shape_1,shape_2)

                        label=label.astype(int)
                        if len(res)>0:
                            row_summary=get_complexity_metrics(data[:num_items],res[:num_items],mod,label[:num_items],explainer,mode=mode)
                        else: 
                            print('Response Length smaller 0 ')
                            number =number+1
                            continue
                        number =number+1
                        '''Savings Section'''
                        if elementwise is not None:
                            df=parameters_to_pandas(explainer)
                            newdf = pd.DataFrame(np.repeat(df.values, len(row_summary), axis=0))
                            newdf.columns = df.columns
                            newdf['explanation'] =np.repeat(str(type(explainer)).split('.')[-1], len(newdf), axis=0)
                            distances_man = pd.concat([row_summary,newdf], axis = 1)
                            distances_man.to_csv(f'{elementwise}/Complexity/{name}_{m}_{str(parameters_to_pandas(explainer).values)}.csv')

                      

                        means = row_summary.mean().add_suffix('_mean')
                        std = row_summary.std().add_suffix('_std')

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

    