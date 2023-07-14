from Benchmarking.metrics.SyntheticEvaluation import Synthetic 
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
from Benchmarking.metrics.synthetic_helper import load_synthetic_data
import numpy as np
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF  import NativeGuideCF
from TSInterpret.InterpretabilityModels.counterfactual.Ates import AtesCF


from RunExplanations import RunExp

if __name__=='__main__':
    explainers=[
        TSEvo(model= None,data=(np.array([0]),[0]), mode = 'feat',backend='PYT',epochs=100),
               # AtesCF(model=None,data=(np.array([0]),[0]),backend='PYT',mode='feat', method= 'opt',max_attempts=1000,
        #max_iter=1000),
                
                                        
                ]

    data_dir=['./Benchmarking/data/multivariate/']
    bm=RunExp(explainers,typ=[''],data_dir=data_dir[0])
    
    bm.calculate_explanations(num_items=20,elementwise='./Results/full/multivariate/elementwise')