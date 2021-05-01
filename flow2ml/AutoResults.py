from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from docx import Document
from docx.shared import Inches

class Auto_Results:
    def __init__(self,model,X_test,Y_test):
        self.model = model
        self.x_test = X_test
        self.prediction = self.model.predict_proba(X_test)
        self.y_test = Y_test
    
    def roc_curve(self,figure_name="roc.jpeg"):
        i= self.prediction.shape
        try:
            if (i[1]>1):
                self.prediction = self.prediction[:, 1]
        except:
            pass
        auc_lr = roc_auc_score(self.y_test, self.prediction)
        fpr_lr, tpr_lr, thresholds_lr = roc_curve(self.y_test, self.prediction)
        plt.figure(figsize=(50, 20))
        plt.plot(fpr_lr, tpr_lr)
        plt.title('ROC Curve', fontsize=70)
        plt.xlabel('False Positive Rate', fontsize=65)
        plt.ylabel('True Positive Rate', fontsize=65)
        plt.savefig(figure_name)
    
    def confusion_matrix(self):
        plt.figure(figsize=(50,20))
        plot_confusion_matrix(self.model,self.x_test,self.y_test)
        plt.title('Confusion Matrix', size = 35)
        plt.savefig('confusion_matrix.jpeg')
    
    def precision_recall_curve(self,figure_name="prc.jpeg"):
        plt.figure(figsize=(50,20))
        plot_precision_recall_curve(self.model,self.x_test,self.y_test)
        plt.title('Precision Recall Curve', size = 35)
        plt.savefig(figure_name)

    def get_results_pdf(self,file_name="report.docx"):
        self.roc_curve()
        self.confusion_matrix()
        self.precision_recall_curve()
        doc = Document()
        doc.add_heading('Model Report')
        doc.add_heading('ROC Curve',level = 2)
        doc.add_paragraph('''
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds.
This curve plots two parameters:
        True Positive Rate
        False Positive Rate
An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives.
The ROC curve for the Input Model is As Follows.
''')
        doc.add_picture('roc.jpeg',width=Inches(6.0))
        doc.add_paragraph('''  ''')
        doc.add_heading('Confusion Matrix', level=2)
        doc.add_paragraph('''
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.
In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm,
typically a supervised learning one (in unsupervised learning it is usually called a matching matrix). Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class (or vice versa).
The name stems from the fact that it makes it easy to see whether the system is confusing two classes (i.e. commonly mislabeling one as another).
It is a special kind of contingency table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the contingency table).
The Confusion Matrix for the Input Model is As shown:       
''')
        doc.add_picture('confusion_matrix.jpeg',width=Inches(5.0))
        doc.add_paragraph('''  ''')
        doc.add_heading("Precision Recall Curve")
        doc.add_paragraph('''
Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.
Precision is a ratio of the number of true positives divided by the sum of the true positives and false positives. It describes how good a model is at predicting the positive class. Precision is referred to as the positive predictive value.
A precision-recall curve is a plot of the precision (y-axis) and the recall (x-axis) for different thresholds, much like the ROC curve.
The Precision Recall Curve for the Input Model is a s shown:

        ''')
        doc.add_picture('prc.jpeg',width=Inches(5.0))
        doc.save(file_name)
#testing codes
'''
df = pd.read_csv('winequality-white.csv', sep=';')
df['quality'] = ['Good' if quality >= 7 else 'Bad' for quality in df['quality']]

from sklearn.model_selection import train_test_split

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

from sklearn.linear_model import LogisticRegression

y_test_int = y_test.replace({'Good': 1, 'Bad': 0})
y_train_int = y_train.replace({'Good': 1,"Bad": 0})
print(y_test_int)
model_lr = LogisticRegression(max_iter=5000).fit(X_train, y_train_int)
auto_res = Auto_Results(model_lr,X_test,y_test_int)
auto_res.get_results_pdf()
'''