import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score, log_loss, accuracy_score
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
        pred = model_logistic_regression.predict(self.val_x)
        evaluateMetricsForRegression(pred)
        
        def evaluateMetricsForRegression(self, pred):
            if (self.algorithms['logistic_regression']['metrics'] is not in ['accuracy', 'recall', 'precision', 'f1_score', 'mae', 'mse', 'rmse', 'r2']):
                raise Exception("Invalid metric to evaluate a regression problem")
             else:
                try:
                    for metric in self.algorithms['logistic_regression']['metrics']:
                        if metric == 'accuracy':
                            model_logistic_regression_accuracy = accuracy_score(self.val_y, pred)
                             print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'model_logistic_regression_accuracy}'
                        if metric == 'recall':
                            model_logistic_regression_recall = recall_score(self.val_y, pred)
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'model_logistic_regression_recall}'
                        if metric == 'precision':
                            model_logistic_regression_precision = precision_score(self.val_y, pred)
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'model_logistic_regression_precision}'
                        if metric == 'f1_score':
                            model_logistic_regression_f1_score = f1_score(self.val_y, pred)
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'model_logistic_regression_f1_score}'
                        if metric == 'log_loss':
                            model_logistic_regression_log_loss = log_loss(self.val_y, pred)
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'model_logistic_regression_log_loss}'
                        if metric == 'mae':
                            mae = metrics.mean_absolute_error(self.val_y, pred)
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'mae}' 
                        if metric == 'mse':
                            mse = metrics.mean_squared_error(self.val_y, pred)
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'mse}' 
                        if metric == 'rmse':
                            rmse = np.sqrt(mse) # or mse**(0.5) 
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'rmse}'
                        if metric == 'r2':
                            r2 = metrics.r2_score(y,yhat)
                            print( f'Mean cross-validated ' + f'{self.algorithms['logistic_regression']['metrics']}' + score of the best_estimator: ' + f'r2}'
                except Exception as e:
                    print(f"evaluation calculation failed due to {e}")
        print(f'Best parameters {model_logistic_regression.best_params_}')

      
        # return the model
        return model_logistic_regression
