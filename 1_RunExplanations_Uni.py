from XTSCBench.metrics.synthetic_helper import get_placeholder_model_and_data
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
from TSInterpret.InterpretabilityModels.leftist.leftist import LEFTIST
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF  import NativeGuideCF
from XTSCBench.metrics.synthetic_helper import get_placeholder_model_and_data
import numpy as np
from RunExplanations import RunExp


if __name__=='__main__':
    model,data= get_placeholder_model_and_data()

    explainers=[
                LEFTIST(None,data=(np.array([[1]]),np.array([[1]])),mode='feat', backend='PYT',learning_process_name='Lime',transform_name='random'),
                LEFTIST(None,data=(np.array([[1]]),np.array([[1]])),mode='feat', backend='PYT',learning_process_name='SHAP',transform_name='random'),
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=False),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),
                
               
                
                
                ]
    



    data_dir=['./XTSCBench/data/univariate']
    bm=RunExp(explainers, typ='_', data_dir=data_dir[0])
    bm.calculate_explanations(num_items=20)


    #Only CNN

    explainers=[
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NUN_CF'),
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='dtw_bary_center'),
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NG') ,                      
                ]

    data_dir=['./XTSCBench/data/univariate']
    bm=RunExp(explainers,classificator=['CNN'], typ='_', data_dir=data_dir[0])
    bm.calculate_explanations(num_items=20)