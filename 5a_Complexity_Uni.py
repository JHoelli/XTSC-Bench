from XTSCBench.ComplexityEvaluation import ComplexityEvaluation
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
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=False),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100, transformer= 'mutate_both'),  
                
               
                
                
                ]


    data_dir=['./XTSCBench/data/univariate']
    bm=ComplexityEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['CNN','LSTM'], data_dir[0],num_items=20,save='./Results/new/univariate/Complexityv1.csv',elementwise='./Results/new/univariate/elementwise',explanation_path='./Results/Explanation/'))


    #Only CNN

    explainers=[
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NUN_CF'),           
                ]
    
    bm=ComplexityEvaluation(None,explainers,(None,[0]))
    print(bm.evaluate_synthetic( '_',['CNN'], data_dir[0],num_items=20,save='./Results/new/univariate/Complexityv2.csv',elementwise='./Results/new/univariate/elementwise',explanation_path='./Results/Explanation/'))