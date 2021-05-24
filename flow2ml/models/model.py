import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import os

class TrainModels:

    def __init__(self,algorithms, train_x, train_y, val_x, val_y) :
        self.algorithms = algorithms
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y

    def applyLogisticRegression(self):
        '''Applies Logistic Regression  with the given metrics and returns the score with best hypertuned params.'''

        logmodel=LogisticRegression()
        # defining hyperparameters to be used for best tuning
        params_grid=[{'penalty' : ['none', 'l1', 'l2', 'elasticnet'] ,
        'C' : np.logspace(-4,4,20),
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter' : [100, 1000, 2500, 5000]
        }
        ]
        # Apply GridSearchCV() to find the best params.
        model_logistic_regression=GridSearchCV(logmodel,param_grid=params_grid, cv=5, verbose=True, n_jobs=-1, scoring=self.algorithms['logistic_regression']['metrics'])
        model_logistic_regression.fit(self.train_x, self.train_y)
        
        print(f'Best parameters {model_logistic_regression.best_params_}')
        print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'{model_logistic_regression.best_score_:.3f}'
              
        # return the model
        return model_logistic_regression
