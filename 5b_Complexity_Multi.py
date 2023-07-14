from Benchmarking.ComplexityEvaluation import ComplexityEvaluation
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
from Benchmarking.metrics.synthetic_helper import load_synthetic_data
from TSInterpret.InterpretabilityModels.leftist.leftist import LEFTIST
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF  import NativeGuideCF
import numpy as np
from Benchmarking.metrics.synthetic_helper import get_placeholder_model_and_data



if __name__=='__main__':
    model,data= get_placeholder_model_and_data()

    explainers=[
               Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='IG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=True),
                #Saliency_PTY(None, None,None, method='DL', mode='feat', tsr=True),
                #Saliency_PTY(None, None,None, method='DLS', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=True),
                #Saliency_PTY(None, None,None, method='SVS', mode='feat', tsr=True),
                #Saliency_PTY(None, None,None, method='FA', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='IG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=False),
                #Saliency_PTY(None, None,None, method='DL', mode='feat', tsr=False),
                #Saliency_PTY(None, None,None, method='DLS', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=False),
                #Saliency_PTY(None, None,None, method='SVS', mode='feat', tsr=True),
                #Saliency_PTY(None, None,None, method='FA', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=False),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),
                #AtesCF(model=None,data=(np.array([0]),[0]),backend='PYT',mode='feat', method= 'opt',max_attempts=1000,     #max_iter=1000),
                
                
                
               
                
                
                
                ]


    data_dir=['./Benchmarking/data/multivariate']
    bm=ComplexityEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['CNN','LSTM'], data_dir[0],num_items=20,save='./Results/new/multivariate/Complexityv1.csv',elementwise='./Results/new/multivariate/elementwise',explanation_path='./Results/Explanation/'))
