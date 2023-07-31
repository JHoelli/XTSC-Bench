from Benchmarking.metrics.SyntheticEvaluation import Synthetic 
from TSInterpret.InterpretabilityModels.Saliency.TSR import TSR, Saliency_PTY
import matplotlib.pyplot as plt
import numpy as np
from Benchmarking.metrics.synthetic_helper import get_placeholder_model_and_data
from Benchmarking.Plots.Plots import     informative_feat_approach_box_plot
#import pandas as pd 
import os


if __name__=='__main__':
    model, data=get_placeholder_model_and_data()
    explainers=[Saliency_PTY(None, None,None, method='GRAD', mode='feat', tsr=True)]

    data_dir=['./Benchmarking/data/univariate/']
    data='univariate'
    if not os.path.isdir(f'./Plots/{data}'):
        os.mkdir(f'./Plots/{data}')
                

    split_column='generation'
    if not os.path.isdir(f'./Plots/{data}/{split_column}'):
        os.mkdir(f'./Plots/{data}/{split_column}')



    split_column='info_feat'
    if not os.path.isdir(f'./Plots/{data}/{split_column}'):
        os.mkdir(f'./Plots/{data}/{split_column}')
    split_column='params'
    if not os.path.isdir(f'./Plots/{data}/{split_column}'):
        os.mkdir(f'./Plots/{data}/{split_column}')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',metric_name='complexity',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Complexity_Double_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',metric_name=['Relevance Rank','Relevance Mass'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Reliability_Double_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Faithfulness_Double_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name=['AverageSensitivity','MaxSensitivity'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Roboustness_Double_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',split_column1='info_feat',split_column2='params_exp', metric_name='complexity',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Complexity_Double2_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',split_column1='info_feat',split_column2='params_exp',metric_name=['Relevance Rank','Relevance Mass'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Reliability_Double2_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex'],split_column1='info_feat',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Double2_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name=['AverageSensitivity','MaxSensitivity'],split_column1='info_feat',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Roboustness_Double2_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',split_column1='Based',split_column2='params_exp', metric_name='complexity',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Complexity_Basis_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',split_column1='Based',split_column2='params_exp',metric_name=['Relevance Rank','Relevance Mass'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Reliability_Basis_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex','monoton_synthetic'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Basis_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Basis_Single_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex','faithfulness_correlation_uniform','faithfulness_correlation_mean'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Basis_uniform_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name=['AverageSensitivity','MaxSensitivity'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Roboustness_Basis_v1.png')
    
    #MODEL SPLIT
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',split_column1='Based',split_column2='params_exp', metric_name='complexity',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Complexity_Model_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',split_column1='Based',split_column2='params_exp',metric_name='Relevance Rank',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Reliability_Model_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name='faithfulness_correlation_synthetic_flex',split_column1='Based',split_column2='params_exp',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Model_Single_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name='AverageSensitivity',split_column1='Based',split_column2='params_exp',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Roboustness_Model_v1.png')
      #TODO Multivariate Roboustness 
    data='multivariate'
    if not os.path.isdir(f'./Plots/{data}'):
        os.mkdir(f'./Plots/{data}')
    data_dir=['./Benchmarking/data/multivariate/']

    if not os.path.isdir(f'./Plots/{data}/{split_column}'):
        os.mkdir(f'./Plots/{data}/{split_column}')
    #informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',metric_name='complexity',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Complexity_Double_v1.png')
    #informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',metric_name=['Relevance Rank','Relevance Mass'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Reliability_Double_v1_{data}.png')
    #informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Faithfulness_Double_v1_{data}.png')
    #informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name=['AverageSensitivity','MaxSensitivity'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/{split_column}/Roboustness_Double_v1_{data}.png')

    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',split_column1='info_feat',split_column2='params_exp', metric_name='complexity',figsize=(3000,600),acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Complexity_Double2_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',split_column1='info_feat',split_column2='params_exp',metric_name=['Relevance Rank','Relevance Mass'],figsize=(3000,600),acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Reliability_Double2_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex'],split_column1='info_feat',split_column2='params_exp',figsize=(3000,600),acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Double2_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name=['AverageSensitivity','MaxSensitivity'],split_column1='info_feat',split_column2='params_exp',figsize=(3000,600),acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Roboustness_Double2_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',split_column1='Based',split_column2='params_exp', metric_name='complexity',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Complexity_Basis_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',split_column1='Based',split_column2='params_exp',metric_name=['Relevance Rank','Relevance Mass'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Reliability_Basis_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex','monoton_synthetic'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Basis_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Basis_Single_v1_{data}.png')
    #informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name=['faithfulness_correlation_synthetic_flex','faithfulness_correlation_uniform','faithfulness_correlation_mean'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Basis_uniform_v1.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name=['AverageSensitivity','MaxSensitivity'],split_column1='Based',split_column2='params_exp',acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Roboustness_Basis_v1_{data}.png')
    
    
     #MODEL SPLIT
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Complexity',split_column1='Based',split_column2='params_exp', metric_name='complexity',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Complexity_Model_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Reliability',split_column1='Based',split_column2='params_exp',metric_name='Relevance Rank',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Reliability_Model_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Faithfulness',metric_name='faithfulness_correlation_synthetic_flex',split_column1='Based',split_column2='params_exp',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Faithfulness_Model_Single_v1_{data}.png')
    informative_feat_approach_box_plot(f'/media/jacqueline/Data/TSInterpret_PIP/Results/new/{data}/elementwise/Roboustness',metric_name='AverageSensitivity',split_column1='Based',split_column2='params_exp',grouping=['CNN','LSTM'],acc_model_threshold= 0.9, with_acc=False,save_path=f'./Plots/{data}/Roboustness_Model_v1_{data}.png')