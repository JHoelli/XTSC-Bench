from Benchmarking.metrics.SyntheticEvaluation import Synthetic 
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
from Benchmarking.metrics.synthetic_helper import load_synthetic_data

from RunExplanations import RunExp


if __name__=='__main__':

    explainers=[Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='IG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=True),
                Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='IG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='GS', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='SG', mode='feat', tsr=False),
                Saliency_PTY(None, None,None, method='FO', mode='feat', tsr=False),
                
                ]


    data_dir=['./Benchmarking/data/multivariate/']

    bm=RunExp(explainers,typ=[''],data_dir=data_dir[0])

    bm.calculate_explanations( elementwise='./Results/full/multivariate/elementwise',num_items=20)
