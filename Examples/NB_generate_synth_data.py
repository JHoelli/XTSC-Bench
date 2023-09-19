from XTSCBench.data.Synthetic import Synthetic

data= Synthetic(10,10, 1, 50, datasetsTypes= ["Middle"],dataGenerationTypes=[None  ], 
                 impTimeSteps=[30],impFeatures=[1],
                 startImpTimeSteps=[10 ], startImpFeatures=[0],
                 models= None)

data.createDatasets('./test_saver/')