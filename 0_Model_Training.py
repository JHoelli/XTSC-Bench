import os 
import numpy as np
from XTSCBench.ClassificationModels.CNN_T import ResNetBaseline, get_all_preds, fit, UCRDataset
from XTSCBench.ClassificationModels.LSTM import LSTM
from XTSCBench.ClassificationModels.TCN import TCN
from XTSCBench.ClassificationModels.Transformer import Transformer
from XTSCBench.ClassificationModels.LSTMWithInputCellAttention import LSTMWithInputCellAttention
import torch 
import sklearn 
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score

def metrics_calc(pred,ground_truth,model,name):
    acc= accuracy_score(ground_truth,pred)
    f1=f1_score(ground_truth,pred,average='micro')
    rec=recall_score(ground_truth,pred)
    pre=precision_score(ground_truth,pred)
    auc=roc_auc_score(ground_truth,pred)
    return pd.DataFrame([[model, name, acc,f1,rec,pre,auc]],columns=['model','name','acc','f1','prec','rec','auc'])





device= 'cpu'
rnn=0.1
hidden_size =10 
n_pred_classes=2

for i in ['multivariate']: 
    #XTSCBench/data/univariate/Training/data
    path= os.listdir(f'./XTSCBench/data/{i}/Training/data')
    for name in path: 
        print(f'{name}')
     #,'multivariate'
        # LOADS THE DATA
        name1= name.replace('Training', 'Testing')
        p1=name.replace('SimulatedTrainingData','SimulatedTrainingMetaData')
        p2=name1.replace('SimulatedTestingData','SimulatedTestingMetaData')
        train_x=np.load(f'./XTSCBench/data/{i}/Training/data/{name}')
        train_y=np.load(f'./XTSCBench/data/{i}/Training/meta/{p1}')[:,0]
        test_x=np.load(f'./XTSCBench/data/{i}/Testing/data/{name1}')
        test_y=np.load(f'./XTSCBench/data/{i}/Testing/meta/{p2}')[:,0]
        train_x_1=train_x.shape[-2]
        train_x_2=train_x.shape[-1]
        #print(f'START {train_x_1} {train_x_2}')

        enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(np.vstack((train_y.reshape(-1,1),test_y.reshape(-1,1))))
        pickle.dump(enc1,open(f'./XTSCBench/ClassificationModels/models/encoder/{name}','wb'))
        train_y=enc1.transform(train_y.reshape(-1,1))
        test_y=enc1.transform(test_y.reshape(-1,1))     

        scaler = sklearn.preprocessing.MinMaxScaler()
        scaler.fit(train_x.reshape(-1,train_x_1*train_x_2))
        train_x = scaler.transform(train_x.reshape(-1,train_x_1*train_x_2)).reshape(-1,train_x_1,train_x_2)
        test_x = scaler.transform(test_x.reshape(-1,train_x_1*train_x_2)) .reshape(-1,train_x_1,train_x_2)                                                       

        train_dataset = UCRDataset(train_x.astype(np.float64),train_y.astype(np.int64))
        test_dataset = UCRDataset(test_x.astype(np.float64),test_y.astype(np.int64))
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

        # THIS SECTION is (Time, Feature)
        if not os.path.isfile( f'./XTSCBench/ClassificationModels/models/LSTM/{name}'):
            # LSTM 
            #input size = # num features
            print(f'LSTM train_x_2 {train_x_2}')
            model = LSTM(train_x_2, hidden_size ,n_pred_classes,rnndropout=rnn).to(device)
            print(f'Input Shape {train_x_1} ,{train_x_2}')
            train_dataset = UCRDataset(train_x.astype(np.float64).reshape(-1,train_x_1,train_x_2),train_y.astype(np.int64))
            test_dataset = UCRDataset(test_x.astype(np.float64).reshape(-1,train_x_1,train_x_2),test_y.astype(np.int64))
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)


            fit(model,train_loader,test_loader)
            if not os.path.isdir('./XTSCBench/ClassificationModels/models/LSTM/'):
                os.mkdir('./XTSCBench/ClassificationModels/models/LSTM/')

            test_preds, ground_truth = get_all_preds(model, test_loader)
            ground_truth=np.argmax(ground_truth,axis=1)
            if os.path.isfile(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv'):
                data_ac =pd.read_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv').drop(columns=['Unnamed: 0'])
                data_ac=pd.concat([data_ac, metrics_calc(test_preds,ground_truth, 'LSTM',name)],ignore_index=True)
            else: 
                data_ac=metrics_calc(test_preds,ground_truth,'LSTM',name)
            data_ac.to_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv')

            torch.save(model, f'./XTSCBench/ClassificationModels/models/LSTM/{name}') # excluded state dict

        #STM with Input Cell 
        if not os.path.isfile( f'./XTSCBench/ClassificationModels/models/LSTM_ATT/{name}'):
            print(f'LSTM  {train_x_2}')
            model=LSTMWithInputCellAttention(train_x_2, 5,2,0.1,10,50).to(device)
            print(f'Input Shape {train_x_1} ,{train_x_2}')
            train_dataset = UCRDataset(train_x.astype(np.float64).reshape(-1,train_x_1,train_x_2),train_y.astype(np.int64))
            test_dataset = UCRDataset(test_x.astype(np.float64).reshape(-1,train_x_1,train_x_2),test_y.astype(np.int64))
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)


            fit(model,train_loader,test_loader)
            if not os.path.isdir('./XTSCBench/ClassificationModels/models/LSTM_ATT/'):
                os.mkdir('./XTSCBench/ClassificationModels/models/LSTM_ATT/')

            test_preds, ground_truth = get_all_preds(model, test_loader)
            ground_truth=np.argmax(ground_truth,axis=1)
            if os.path.isfile(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv'):
                data_ac =pd.read_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv').drop(columns=['Unnamed: 0'])
                data_ac=pd.concat([data_ac, metrics_calc(test_preds,ground_truth, 'LSTM_ATT',name)],ignore_index=True)
            else: 
                data_ac=metrics_calc(test_preds,ground_truth,'LSTM_ATT',name)
            data_ac.to_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv')

            torch.save(model, f'./XTSCBench/ClassificationModels/models/LSTM_ATT/{name}')#exclude state dict



        # THIS SECTION is (Feature , Time)

        # CNN 
        if not os.path.isfile( f'./XTSCBench/ClassificationModels/models/CNN/{name}'):
            print(f'CNN  {train_x_2}')
            model = ResNetBaseline(in_channels=train_x_2, num_pred_classes=n_pred_classes)
            train_dataset = UCRDataset(train_x.astype(np.float64).reshape(-1,train_x_2,train_x_1),train_y.astype(np.int64))
            test_dataset = UCRDataset(test_x.astype(np.float64).reshape(-1,train_x_2,train_x_1),test_y.astype(np.int64))
            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

            fit(model,train_loader,test_loader)
            if not os.path.isdir('./XTSCBench/ClassificationModels/models/CNN/'):
                os.mkdir('./XTSCBench/ClassificationModels/models/CNN/')

            test_preds, ground_truth = get_all_preds(model, test_loader)
            ground_truth=np.argmax(ground_truth,axis=1)
            if os.path.isfile(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv'):
                data_ac =pd.read_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv').drop(columns=['Unnamed: 0'])
                data_ac=pd.concat([data_ac, metrics_calc(test_preds,ground_truth, 'CNN',name)],ignore_index=True)
            else: 
                data_ac=metrics_calc(test_preds,ground_truth,'CNN',name)
            data_ac.to_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv')

            torch.save(model, f'./XTSCBench/ClassificationModels/models/CNN/{name}')#exclude state dict


        #
        #Transformer 
        #TODO
        #Input Size, Sequence Length
        #model=Transformer(train_x_2,train_x_1, 6, 5, 0.1,2,time=50)

        #fit(model,train_loader,test_loader)
        #if not os.path.isdir('./XTSCBench/ClassificationModels/models/Trans/'):
        #    os.mkdir('./XTSCBench/ClassificationModels/models/Trans/')

        #test_preds, ground_truth = get_all_preds(model, test_loader)
        #ground_truth=np.argmax(ground_truth,axis=1)
        #if os.path.isfile(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv'):
        #    data_ac =pd.read_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv').drop(columns=['Unnamed: 0'])
        #    data_ac=pd.concat([data_ac, metrics_calc(test_preds,ground_truth, 'Trans',name)],ignore_index=True)
        #else: 
        #    data_ac=metrics_calc(test_preds,ground_truth,'Trans',name)
        #data_ac.to_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv')

        #torch.save(model, f'./XTSCBench/ClassificationModels/models/Trans/{name}')#exclude state dict

                
        # TCN
#        num_chans = [5] * (3 - 1) + [train_x_2]
#        net=TCN(train_x_2,2,num_chans,4,0.1,time=train_x_1)

 #       fit(model,train_loader,test_loader)
 #       if not os.path.isdir('./XTSCBench/ClassificationModels/models/TCN/'):
 #           os.mkdir('./XTSCBench/ClassificationModels/models/TCN/')
#
        #test_preds, ground_truth = get_all_preds(model, test_loader)
        #ground_truth=np.argmax(ground_truth,axis=1)
        #if os.path.isfile(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv'):
        #    data_ac =pd.read_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv').drop(columns=['Unnamed: 0'])
        #    data_ac=pd.concat([data_ac, metrics_calc(test_preds,ground_truth, 'TCN',name)],ignore_index=True)
        #else: 
        #    data_ac=metrics_calc(test_preds,ground_truth,'TCN',name)
        #data_ac.to_csv(f'./XTSCBench/ClassificationModels/models/preformance_{i}.csv')
        #torch.save(model, f'./XTSCBench/ClassificationModels/models/TCN/{name}')#exclude state dict
    