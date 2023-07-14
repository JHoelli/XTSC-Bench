
#from turtle import pd
from sklearn.metrics import precision_recall_fscore_support,classification_report, accuracy_score, f1_score , recall_score, precision_score
import numpy as np
from XTSC-Bench.Evaluation import Evaluation
import pandas as pd
import torch

def accuracy(original,pred):
    if len(original.shape)>1:
        original=np.argmax(original, axis = 1)
    if len(pred.shape)>1:
        pred = np.argmax(pred, axis=1) 
    return accuracy_score(original , pred)

def f(original,pred):
    if len(original.shape)>1:
        original=np.argmax(original, axis = 1)
    if len(pred.shape)>1:
        pred = np.argmax(pred, axis=1)
    return f1_score(original , pred)

def recall(original,pred):
    if len(original.shape)>1:
        original=np.argmax(original, axis = 1)
    if len(pred.shape)>1:
        pred = np.argmax(pred, axis=1)
    return precision_score(original , pred)

def precision(original,pred):
    return recall_score(original , pred)

def trust_score():
    '''https://docs.seldon.io/projects/alibi/en/stable/methods/TrustScores.html'''
    pass

def linearity():
    '''https://docs.seldon.io/projects/alibi/en/stable/methods/LinearityMeasure.html'''
    pass

def _get_classification_metrics(original, pred):
    if len(original.shape)>1:
        original=np.argmax(original, axis = 1)
    if len(pred.shape)>1:
        pred = np.argmax(pred, axis=1)

    acc = accuracy(original,pred)
    f1 = f(original,pred)
    prec = precision(original,pred)
    rec = recall(original,pred)

    return [[acc, f1, prec, rec]]

class Classification_metrics(Evaluation):
    """
    Calculates the L0, L1, L2, and L-infty distance measures.
    """

    def __init__(self, mlmodel):
        super().__init__(mlmodel)
        self.columns = ["accuracy", "f1", "precision", "recall"]

    def get_evaluation(self, x,y):
        x =torch.from_numpy(x).float()
        pred= self.mlmodel(x).detach().numpy()

        distances = _get_classification_metrics(y,pred)

        return pd.DataFrame(distances, columns=self.columns)