
from XTSCBench.FaithfulnessEvaluation import FaithfulnessEvaluation
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
from XTSCBench.metrics.synthetic_helper import load_synthetic_data
from TSInterpret.InterpretabilityModels.leftist.leftist import LEFTIST
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF  import NativeGuideCF
import numpy as np
from XTSCBench.metrics.synthetic_helper import get_placeholder_model_and_data

if __name__=='__main__':

    model,data= get_placeholder_model_and_data()
    explainers=[         
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),              
                ]
    

    model,data= get_placeholder_model_and_data()

    data_dir=['./XTSCBench/data/univariate']
    bm=FaithfulnessEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( 'SimulatedTestingData_Middle_AutoRegressive_F_1_TS_50',['LSTM','CNN'], data_dir[0],num_items=2,save='./Results/univariate/Faithufulness2.csv',elementwise='./Results/univariate/elementwise',explanation_path='./Results/Explanation/'))

