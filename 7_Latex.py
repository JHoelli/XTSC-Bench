from Benchmarking.metrics.SyntheticEvaluation import Synthetic 
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
from Benchmarking.metrics.synthetic_helper import load_synthetic_data
import numpy as np
from TSInterpret.InterpretabilityModels.counterfactual.TSEvoCF import TSEvo
from TSInterpret.InterpretabilityModels.counterfactual.NativeGuideCF  import NativeGuideCF
from Benchmarking.metrics.synthetic_helper import get_placeholder_model_and_data
from Benchmarking.Plots.Plots import  Latex_Table,   box_plot_itemwise_data, plot_itemwise_ORG_vs_GT,informative_feat_approach_box_plot
from TSInterpret.InterpretabilityModels.leftist.leftist import LEFTIST
#import pandas as pd 
import os


if __name__=='__main__':
    #Results/new/univariate/elementwise
    print(Latex_Table('./Results/new/univariate/elementwise'))
    print(Latex_Table('./Results/new/multivariate/elementwise'))