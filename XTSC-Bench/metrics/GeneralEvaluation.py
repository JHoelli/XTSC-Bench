from XTSC-Bench.Evaluation import Evaluation
import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd 
import numpy as np
from XTSC-Bench.metrics.synthetic_helper import get_preds,load_synthetic_data,manipulate_exp_method,scaling, get_explanation,does_entry_already_exist
import os
from XTSC-Bench.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from XTSC-Bench.metrics.general_metrics import get_general_metrics
import quantus

class GeneraEvaluation(Evaluation):
    """
    #TODO Absprung in Quantus
 """

    def __init__(self,  mlmodel,explainer,metrics,data=None):
        super().__init__(mlmodel)
        self.models=mlmodel
        self.explainers=explainer
        #TODO 

        self.metrics=metrics

        #TODO IS THIS removable ? 
        self.data=data
        self.columns=[]


    def evaluate(self, items,interpretation):
        pass

    def evaluate_synthetic(self,types, classificator, data_dir, num_items=100,save=None,elementwise=None, explanation_path=None):
        #TODO START WITH THIS AND QUANTUS ? 
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
                            number =number+1
                            continue  
                        '''Load Model and Manipulate Explainer'''
                        mod= torch.load(f'./XTSC-Bench/ClassificationModels/models_new/{m}/{modelName}',map_location='cpu')
                        mname=name.replace('Testing','Training')
                        explainer = manipulate_exp_method(d_train, l_train, shape_1, shape_2, scaler, explainer, mod)

                        if type(explainer) ==str: 
                            #TODO add Log?
                            print('Predictor returns constant predictoe')
                            continue
                        
                        '''Calculate Explanations'''
                        y_pred=[]
                        res=[]
                        if explanation_path is None:
                            #TODO NOT TESTED                             
                            for x,y in zip(data[:num_items],label[:num_items]):
                                if 'CNN' in str(type(mod)):
                                    x =torch.from_numpy(x).float().reshape(1,shape_2,shape_1)
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
                            if 'CNN' in str(type(mod)):
                                res=res.reshape(-1,shape_2,shape_1)
                            else:
                                res=res.reshape(-1,shape_1,shape_2)
                            y_pred = get_preds(mod, res)

                        if 'CNN' in str(type(mod)):
                            mode='feat'
                            data = data.reshape(-1,shape_2,shape_1)
                            res=res.reshape(-1,shape_2,shape_1)
                            print(data.shape)
                            print(res.shape)
                        else:
                            mode='time'
                            data = data.reshape(-1,1,shape_1,shape_2)
                            res=res.reshape(-1,1,shape_1,shape_2)
                            print(data.shape)
                            print(res.shape)
                        #else:

                        #TODO Call to metric ! 
                        #TODO Adapt perturbation and similarity function 

                        #def callable(model,inputs,targets, device='cpu'):
                        #    explainer.model=model
                            #print(inputs.shape)
                            #print(targets.shape)
                        #    res=[]
                        #    for x1, y1 in zip(inputs,targets):
                                #x =torch.from_numpy(x1).float().reshape(1,inputs.shape[-2],inputs.shape[-1])
                                #pred= torch.nn.functional.softmax(model(x)).detach().numpy()
                        #        res.append(explainer.explain(x1.reshape(-1,inputs.shape[-2],inputs.shape[-1]), int(y1)))
                          
                        #    return res
                        label=label.astype(int)
                        row_summary=get_general_metrics(data[:num_items],res[:num_items],mod,label[:num_items],explainer,mode=mode)

                       # print(scores)

                        #import sys
                        #sys.exit(1)
                    

                        '''Savings Section'''
                        if elementwise is not None:
                            df=parameters_to_pandas(explainer)
                            newdf = pd.DataFrame(np.repeat(df.values, len(row_summary), axis=0))
                            newdf.columns = df.columns
                            newdf['explanation'] =np.repeat(str(type(explainer)).split('.')[-1], len(newdf), axis=0)
                            distances_man = pd.concat([row_summary,newdf], axis = 1)
                            distances_man.to_csv(f'{elementwise}/{name}_{m}_{str(parameters_to_pandas(explainer).values)}.csv')
                        #print(distances_man.head())
                      

                        means = row_summary.mean().add_suffix('_mean')
                        std = row_summary.std().add_suffix('_std')
                        #print('Means')
                        #print(means)

                        #print('STD')
                        #print(std)
                        #mydf.loc['newindex'] = myseries
                        su =pd.DataFrame([modelName,typ,generation,m], columns=SummaryTableCol)
                        new= pd.concat([means,std]).to_frame().T#pd.concat([means, std],axis=1).T
                        #print(new)
                        new_row_summary=pd.concat([new,parameters_to_pandas(explainer)], axis = 1)
                        new_row_summary= pd.concat([new_row_summary, su], axis = 1)
                        SummaryTable= pd.concat([new_row_summary,SummaryTable],ignore_index=True)

                        #print(SummaryTable)

                        #import sys 
                        #s#ys.exit(1)


                        if save is None:
                            SummaryTable.to_csv('temp.csv')
                        else: 
                            SummaryTable.to_csv(f'{save}')
                        #print(row_summary)

    