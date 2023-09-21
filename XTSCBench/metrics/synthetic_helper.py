import os 
import numpy as np 
import torch
from torch import nn
import torch.nn.functional as F 
from XTSCBench.ClassificationModels.CNN_T import ResNetBaseline
from XTSCBench.metrics.metrics_helper import parameters_to_pandas, new_kwargs
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import torch.utils.data as data_utils
import copy
from torch.utils.data import DataLoader


def scaling(data, label, data_shape_1, data_shape_2):
        '''
        Helper Function to Scale the synthetic data.
        Attributes: 
            data np.array: data to be scaled
            label np.array: labels for the data 
            data_shape_1 int: shape of the first data dimension
            data_shape_2 int: shape of the second data dimension
        Returns: 
            np.array, torch.Dataset, sklearn.preprocessimg.MinMacScaler: Returns scaled data, torch data loader and the scaler.
        '''
        data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        dataRNN = data.reshape(data.shape[0] ,data_shape_1,data_shape_2)
        test_dataRNN = data_utils.TensorDataset(torch.from_numpy(dataRNN),torch.from_numpy(label))
        test_loaderRNN = data_utils.DataLoader(test_dataRNN, batch_size=1, shuffle=False)
        return dataRNN,test_loaderRNN,scaler


def does_entry_already_exist(old_data, m, generation, typ, modelName):
    if old_data is not None:
        #['Datasets','Typ','Generation','models'
        flag=old_data[old_data['models']==m]
        if len(flag)!=0:
            ##print(f'Model Found {m}')
            flag= flag[flag['Generation']==generation]
            if len(flag)!=0:
                ##print(f'Generation Found {generation}')
                flag= flag[flag['Typ']==typ]
                if len(flag)!=0:
                    ##print(f'Typ Found {typ}')
                    flag= flag[flag['Datasets']==modelName]
                    if len(flag)!=0:
                        ##print(f'Data Found {modelName}')
                        #TODO ADD PARAM SPECIFICATION
                        return True
    return False

def manipulate_exp_method( d_train, l_train, data_shape_1, data_shape_2,scaler, saliency, mod, check_consist=False):
        '''
        Manipulate Dict Key of Explanation Instance for current iteration 
        Attributes: 
            d_train np.array: data 
            l_train np.array: labels for data 
            data_shape_1: shape of the original data in dimension 1 
            data_shape_2: shape of the original data in dimension 2 
            scaler Callable: Scaling Instance 
            saliency Callable: Explainer Instance to be manipulated
            mod Callable: Model 
        Returns 
            Callable: the newly manipulated explainer Instance
        
        '''
        di=new_kwargs(saliency)

        di['model']=mod

        if 'CNN' in str(type(mod)):
            di['mode'] ='feat'
            if 'NumFeatures' in di.keys():
                di['NumFeatures']=data_shape_2
                di['NumTimeSteps']=data_shape_1
        else: 
            #print('!!!LSTM')
            di['mode'] ='time'
            if 'NumFeatures' in di.keys():
                di['NumFeatures']=data_shape_2
                di['NumTimeSteps']=data_shape_1
        d_train = d_train.reshape(d_train.shape[0],data_shape_1*data_shape_2)
        d_train = scaler.transform(d_train)
        d_train=d_train.reshape(d_train.shape[0],data_shape_1,data_shape_2) 
        dataRNN_train=d_train
        if 'CNN' in str(type(mod)):
            dataRNN_train = np.swapaxes(d_train,-1,-2)
            print('CNN', d_train.shape)
            print('CNN2', dataRNN_train.shape)

        di_old=copy.deepcopy(di)

        '''  Test of Prediction'''
        item = torch.from_numpy(dataRNN_train)
        out = mod(item.float())
        y_pred = torch.nn.functional.softmax(out).detach().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        equal=  all(element == y_pred[0] for element in y_pred)
        if equal and check_consist: 
            print ("Prediction of Classifier is constant. --> Calculation not possible")
            return "Prediction of Classifier is constant."                 
        try:
            di['data'] =(dataRNN_train,l_train.reshape(-1).astype(np.int64))
            print('dataRNN',dataRNN_train.shape)

            

            saliency = type(saliency)(**di)     

        except:

            saliency = type(saliency)(**di_old)
       
        return saliency

   
def get_preds( mod, cfs):
        '''
        #TODO eventuell sind Label notwendig
        '''
        mod.eval()
        cfs=cfs.reshape(-1, cfs.shape[-2], cfs.shape[-1])
        loader =  DataLoader(cfs, batch_size=64, shuffle=True)
        with torch.no_grad():
            all_preds = []
            #labels = []
            for batch in loader:
                item = batch
                preds = mod(item.float())
                all_preds = all_preds + preds.argmax(dim=1).tolist()
        return all_preds


def get_explanation( data, label, data_shape_1, data_shape_2, saliency, mod,mode):
        '''
        Calculates the explanations for the given data. 
         Attributes: 
            data np.array: data 
            label np.array: labels for data 
            data_shape_1: shape of the original data in dimension 1 
            data_shape_2: shape of the original data in dimension 2 
            saliency Callable: Explainer Instance to be manipulated
            mod Callable: Model 
        Returns 
            List of explanations

        '''
        sal=[]
        for x1,y1 in zip(data,label):
            x1=np.array(x1)
            if mode == 'feat':
                x1=np.swapaxes(x1,-1,-2)
                print('x1',x1.shape)
                x =torch.from_numpy(x1).float().reshape(1,data_shape_2,data_shape_1)
                print('x',x.shape)
            else: 
                x =torch.from_numpy(x1).float().reshape(1,data_shape_1,data_shape_2)
            if 'counterfactual' in str(type(saliency)):
                pred= torch.nn.functional.softmax(mod(x)).detach().numpy()
                res,label_cf=saliency.explain(x.detach().numpy(),np.array([np.argmax(pred[0])]))
                res= res.reshape(-1,res.shape[-2],res.shape[-1])


            else:
                res=np.array(saliency.explain(x, int(y1)))
                res= res.reshape(-1,res.shape[-2],res.shape[-1])
            if mode=='feat':
                res=np.swapaxes(np.array(res),-1,-2)
            if res is not None:
                sal.append(res.tolist()[0])

            else:
                if 'CNN' in str(type(mod)):
                    sal.append(np.full((1,data_shape_2,data_shape_1)[0], None).tolist())#np.ones_like(x1.reshape(data_shape_2,data_shape_1)))
                else:
                    sal.append(np.full((1,data_shape_1,data_shape_2)[0], None).tolist())#np.ones_like(x1.reshape(data_shape_1,data_shape_2)))
        print('SAL', np.array(sal).shape)
        return sal


def get_placeholder_model_and_data():
    '''
    Placeholder Model and Data for Explainers that need it in init.
    Returns: 
        torch.nn.Module, np.array: Placeholder Model and Data
    '''
    model=ResNetBaseline(in_channels=1,num_pred_classes=2)
    data=np.ones((1,1,100))
    return model, data

def load_synthetic_data(keywords, univariate, return_train= False):
    '''
    Loads Synthetic Data all, or filtered for keywords. 

    Arguments:
        keywords list: List of Keywotds to filter datasets
        data_dir str: Path to saved data

    Returns: 
        list,list,list: Thripel consisting of testing data, metadata and label

    
    '''
    data={}
    meta={}
    label={}
    data_train={}
    meta_train={}
    label_train={}

    for f in keywords: 
        for path in os.listdir(f'{univariate}/Testing/data'):
            if f in path:
                ##print(True)
                data[path] = np.load(f'{univariate}/Testing/data/{path}', allow_pickle=True)
                p=path.replace('SimulatedTestingData','SimulatedTestingMetaData') 
                meta[path]=np.load(f'{univariate}/Testing/meta/{p}',allow_pickle=True)
                label[path]= meta[path][:,0]
                meta[path] =meta[path]
                if return_train:
                    a=path.replace('Testing','Training')
                    data_train[path] = np.load(f'{univariate}/Training/data/{a}', allow_pickle=True)
                    p=path.replace('SimulatedTestingData','SimulatedTrainingMetaData')
                    meta_train[path]=np.load(f'{univariate}/Training/meta/{p}', allow_pickle=True)
                    label_train[path]= meta_train[path][:,0]
                    meta_train[path] =meta_train[path]
    if not return_train:
        return data, meta, label
    else:
        return data_train, meta_train, label_train, data, meta, label