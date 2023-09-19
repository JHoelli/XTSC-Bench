# XTSC-Bench: A Benchmarking Tool for Time Series Explanation Alorithms 

XTSC-Bench is a python package to faciliate the benchmarking of explanation algorithms of Time Series Classifiers. The intention is to provide a tool to the research community to enable an easy evaluation of newly develped algorithms and therby encouraging progess in this research area. XTSC-Bench consists of synthetic data with ground truth and already trained models for a standardize evaluation, as well as metrics to measuere the performance of the evaluated Approaches. The Core Metric is thereby provided by Quantos, XTSC-Bench provides helper function as well as metric adaptions to Quantos to make the metrics useable on TSC explainers. 
Further, it is compatible with TSInterpret and provides therefore easy access to baselines. 


## Installation
Clone the Github Repository for full availability of the Synthetic Datasets. 
```
git clone [TO BE SPECIFIED]
```

Install the Benchmarking tool by using pip. (In Future, we will also make the Tool available via pip.)

```
pip install .
```



## Quickstart
The following example Benchmarks GS-TSR [1] against GRAD-TSR [1] on the Complexity Metric. 
[1] Ismail, Aya Abdelsalam, et al. "Benchmarking deep learning interpretability in time series predictions." Advances in neural information processing systems 33 (2020): 6441-6452.

### Import 
```
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import torch 
from Benchmarking.ClassificationModels.CNN_T import ResNetBaseline, UCRDataset,fit
from Benchmarking.ClassificationModels.LSTM import LSTM
from Benchmarking.ComplexityEvaluation import ComplexityEvaluation
from tslearn.datasets import UCR_UEA_datasets
import sklearn
import numpy as np 

```

### With Synthetic Data 
Choose Explainer to Evalaute:
```python
 model,data= get_placeholder_model_and_data()

    explainers=[
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=True),             
                ]

```
For Evalaution on the standardized Synthetic Data with the already trained models:    
```python
    data_dir=['./Benchmarking/data/univariate']
    bm=ComplexityEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['CNN','LSTM'], data_dir[0],num_items=20,save='./Results/new/univariate/Complexityv1.csv',elementwise='./Results/new/univariate/elementwise',explanation_path='./Results/Explanation/'))
```

### With Custom Data
Load the Data and Train a Model
```python
dataset='ECG5000'
train_x,train_y, test_x, test_y=UCR_UEA_datasets().load_dataset(dataset)

enc1=sklearn.preprocessing.OneHotEncoder(sparse=False).fit(np.vstack((train_y.reshape(-1,1),test_y.reshape(-1,1))))
train_y=enc1.transform(train_y.reshape(-1,1))
test_y=enc1.transform(test_y.reshape(-1,1))    

n_pred_classes =train_y.shape[1]
train_dataset = UCRDataset(train_x.astype(np.float64),train_y.astype(np.int64))
test_dataset = UCRDataset(test_x.astype(np.float64),test_y.astype(np.int64))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
model =LSTM(1, 10 ,n_pred_classes,rnndropout=0.1).to('cpu') #ResNetBaseline(in_channels=1, num_pred_classes=n_pred_classes)
fit(model,train_loader,test_loader)
model.eval()
```
Select Explainer to Benchmark and Metrics
```python
explainer =  [Saliency_PTY(model, 140,1, method='GRAD', mode='time', tsr=True),Saliency_PTY(model, 140,1, method='GS', mode='time', tsr=True)]
bm=ComplexityEvaluation(explainer=explainer)
print(bm.evaluate(test_x[0:2], np.argmax(test_y[0:2],axis=1),model, mode='time',aggregate=True))

```

### With a Custom Explainer Model
If you want to benchmark your own custom implemented explanation approach. It shoud follow the following interface:
```python
class ExplainerModel():
    def __init__(self,ml_model,mode):
        '''
        ml_model torch.nn.module: Classification Model
        mode str: 'time' if the first dimension is time, 'feat' otherwise 
        '''
        self.model = ml_model
        self.mode = mode
    
    def explain(self,x,y) --> np.array :
        '''
        x np.array: instance to explain in shape (1,feat,time) or (1,time,feat) depending on Mode
        y int: Label of instance
        '''
        pass

```
Additional Parameters can be added to your liking in the `init`. 
### Additional Examples
More examples can be found in ./Examples. 

## Replicating Paper Results
The Scripts and original code used to obtain the results from the paper can be found in the branch [results](https://github.com/JHoelli/XTSC-Bench/tree/results).
