from XTSCBench.RoboustnessEvaluation import RoboustnessEvaluation
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
import numpy as np
from XTSCBench.metrics.synthetic_helper import get_placeholder_model_and_data

if __name__=='__main__':
    model,data= get_placeholder_model_and_data()

    explainers=[
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='IG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='IG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=False),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),
                #AtesCF(model=None,data=(np.array([0]),[0]),backend='PYT',mode='feat', method= 'opt',max_attempts=1000,     #max_iter=1000),
                
                
                
               
                        
                ]
    
    model,data= get_placeholder_model_and_data()

    data_dir=['./XTSCBench/data/multivariate']
    bm=RoboustnessEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['LSTM','CNN'], data_dir[0],num_items=20,save='./Results/new/multivariate/Robustnessv1.csv',elementwise='./Results/new/multivariate/elementwise',explanation_path='./Results/Explanation/'))


