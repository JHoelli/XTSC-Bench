from XTSCBench.ComplexityEvaluation import ComplexityEvaluation
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
import numpy as np
from XTSCBench.metrics.synthetic_helper import get_placeholder_model_and_data



if __name__=='__main__':
    model,data= get_placeholder_model_and_data()

    explainers=[
               Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='IG', mode='feat', tsr=True),
                ]


    data_dir=['./XTSCBench/data/univariate']
    bm=ComplexityEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['CNN'], data_dir[0],num_items=2,save='./Results/multivariate/Complexityv1.csv',elementwise='./Results/multivariate/elementwise',explanation_path='./Results/Explanation/'))