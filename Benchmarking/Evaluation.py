from abc import ABC, abstractmethod

import pandas as pd


class Evaluation(ABC):
    def __init__(self, mlmodel, hyperparameters: dict = None):
        """
        Parameters
        ----------
        mlmodel:
            Classification model. (optional)
        hyperparameters:
            Dictionary with hyperparameters, could be used to pass other things. (optional)
        """
        self.mlmodel = mlmodel

    @abstractmethod
    def evaluate(
        self, items,  interpretation
    ) -> pd.DataFrame:
        """Compute evaluation measure"""

    @abstractmethod
    def evaluate_synthetic(
        self, items,  interpretation
    ) -> pd.DataFrame:
        """Compute evaluation measure"""