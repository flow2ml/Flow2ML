import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
from sklearn.linear_model import LogisticRegression
from flow2ml import Auto_Results
from sklearn.model_selection import train_test_split

df = pd.read_csv('winequality-white.csv', sep=';')
df['quality'] = [1 if quality >= 7 else 0 for quality in df['quality']]
X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model_lr = LogisticRegression(max_iter=5000).fit(X_train, y_train)
model_lr.fit(X_train,y_train)
auto_res = Auto_Results(model_lr,X_test,y_test)
auto_res.get_results_docx()