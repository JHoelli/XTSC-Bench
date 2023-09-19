from XTSCBench.data.Synthetic import Synthetic
from XTSCBench.metrics.synthetic_helper import load_synthetic_data

data= Synthetic(10,10, 1, 50, datasetsTypes= ["Middle"],dataGenerationTypes=[None  ], 
                 impTimeSteps=[30],impFeatures=[1],
                 startImpTimeSteps=[10 ], startImpFeatures=[0],
                 models= None)

data.createDatasets('./test_saver/')

# Loading the data 
load_synthetic_data([],'./test_saver/' , return_train= False)