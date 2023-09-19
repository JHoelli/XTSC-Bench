from XTSCBench.RobustnessEvaluation import RobustnessEvaluation
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
import numpy as np
from XTSCBench.metrics.synthetic_helper import get_placeholder_model_and_data

if __name__=='__main__':
    model,data= get_placeholder_model_and_data()

    explainers=[
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),            
                
                
               
                        
                ]
    
    model,data= get_placeholder_model_and_data()

    data_dir=['./XTSCBench/data/multivariate']
    bm=RobustnessEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['LSTM','CNN'], data_dir[0],num_items=1,save='./Results/multivariate/Robustnessv1.csv',elementwise='./Results/multivariate/elementwise',explanation_path='./Results/Explanation/'))