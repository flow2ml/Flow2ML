import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import os

class Apply_Models:

    def __init__(self,algorithms) :
        self.algorithms = algorithms

    def applyLogisticRegression(self,model):
        '''Applies Logistic Regression to the model with the given activation function and metrics and returns the score with best hypertuned params.'''

        logmodel=LogisticRegression(model,self.algorithms['logistic_regression']['activation_function'],self.algorithms['logistic_regression']['metrics'])
        # defining hyperparameters to be used for best tuning
        params_grid=[{'penalty' : ['none', 'l1', 'l2', 'elasticnet'] ,
        'C' : np.logspace(-4,4,20),
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter' : [100, 1000, 2500, 5000]
        }
        ]
        # Apply GridSearchCV() to find the best params.
        grided=GridSearchCV(logmodel,param_grid=params_grid, cv=3, verbose=True, n_jobs=-1)
        #Build model using best params on test_data
        best_fit=grided.fit(logmodel)
        best_fit.best_estimator_
        # return the score 
        return best_fit.score(logmodel)
