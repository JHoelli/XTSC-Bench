from Benchmarking.metrics.SyntheticEvaluation import Synthetic 
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
from Benchmarking.metrics.synthetic_helper import load_synthetic_data
import numpy as np
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF  import NativeGuideCF
from Benchmarking.metrics.synthetic_helper import get_placeholder_model_and_data
from RunExplanations import RunExp
#import pandas as pd 


if __name__=='__main__':
    model, data=get_placeholder_model_and_data()
    explainers=[NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NUN_CF'),
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='dtw_bary_center'),
                NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NG') ,
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100, transformer= 'mutate_both'),                      
                ]

    data_dir=['./Benchmarking/data/univariate/']
    bm=RunExp(explainers,typ=[''],data_dir=data_dir[0],classificator=['CNN'])
    bm.calculate_explanations( elementwise='./Results/full/univariate/elementwise')

    explainers=[
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),
                TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100, transformer= 'mutate_both'),                      
                ]

    bm=RunExp(explainers,typ=[''],data_dir=data_dir[0],classificator=['LSTM'])
    bm.calculate_explanations( elementwise='./Results/full/univariate/elementwise')