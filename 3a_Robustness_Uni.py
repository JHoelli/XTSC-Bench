from XTSCBench.RoboustnessEvaluation import RoboustnessEvaluation
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
            LEFTIST(None,data=(np.array([[1]]),np.array([[1]])),mode='feat', backend='PYT',learning_process_name='Lime',transform_name='random'),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),                
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=False),

               
                
                
                ]
    

    model,data= get_placeholder_model_and_data()

    data_dir=['./XTSCBench/data/univariate']
    bm=RoboustnessEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['LSTM','CNN'], data_dir[0],num_items=20,save='./Results/new/univariate/Roboustnessv2.csv',elementwise='./Results/new/univariate/elementwise',explanation_path='./Results/Explanation/'))


    #Only CNN

    explainers=[
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NUN_CF'),
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='dtw_bary_center'),
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NG') ,                
                ]

    bm=RoboustnessEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['CNN'], data_dir[0],num_items=20,save='./Results/new/univariate/Roboustnessv2.csv',elementwise='./Results/new/univariate/elementwise',explanation_path='./Results/Explanation/'))