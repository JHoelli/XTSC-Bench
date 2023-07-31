from Benchmarking.metrics.SyntheticEvaluation import Synthetic 
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
from Benchmarking.metrics.synthetic_helper import load_synthetic_data
import numpy as np
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF  import NativeGuideCF
from Benchmarking.metrics.synthetic_helper import get_placeholder_model_and_data
from Benchmarking.Plots.Plots import      plot_itemwise_ORG_vs_GT
from TSInterpret.InterpretabilityModels.leftist.leftist import LEFTIST
#import pandas as pd 
import os


if __name__=='__main__':
    model, data=get_placeholder_model_and_data()
    #explainers=[NativeGuideCF(model=model,data=(data,[0]), backend='PYT', mode='feat',method='NUN_CF')   ]
    explainers=[Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True)]

    data_dir=['./Benchmarking/data/univariate/']
    #ata_name,data_dir,model, explainer,i,resultpath,save_path

    plot_itemwise_ORG_vs_GT('SimulatedTrainingData_RareTime_AutoRegressive_F_1_TS_50.npy',data_dir[0],'CNN', explainers[0],0,resultpath='./', save_path='./Plots/SingleItem1.png',roboust=True)
    explainers=[LEFTIST(None,data=(np.array([[1]]),np.array([[1]])),mode='feat', backend='PYT',learning_process_name='Lime',transform_name='random'),]


    plot_itemwise_ORG_vs_GT('SimulatedTrainingData_RareTime_AutoRegressive_F_1_TS_50.npy',data_dir[0],'CNN', explainers[0],0,resultpath='./', save_path='./Plots/SingleItem2.png',roboust=True)
