# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 13:28:57 2022

@author: avidvans3
"""

"LDA QDA and Naive Bayes classifier and comparison"

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
data=pd.read_csv('iris.csv')
import seaborn as sns

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Y=data['class']
# for i in range(0,len(Y)):
#     if Y[i]=='Iris-setosa':
#         Y[i]=1
#     if Y[i]=='Iris-versicolor':
#         Y[i]=2
#     if Y[i]=='Iris-virginica':
#         Y[i]=3
    
data=data.drop('class',axis=1)
X=data;


clf = LinearDiscriminantAnalysis()
clf.fit(X, Y)

print(clf.score(X,Y))
y_LDA=clf.predict(X)
y1=clf.predict(np.reshape([1,3,5,6],(1, -1)))

conlda=confusion_matrix(Y,y_LDA)
disp=ConfusionMatrixDisplay(confusion_matrix=conlda, display_labels=["iris-setosa", "Iris-versicolor", "Iris-virginica"])

disp.plot()

plt.figure()
sns.countplot(x=Y)

"Lets do QDA"

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, Y)
print(clf.score(X,Y))

y_qda=clf.predict(X)
conqda=confusion_matrix(Y,y_qda)
disp=ConfusionMatrixDisplay(confusion_matrix=conqda, display_labels=["iris-setosa", "Iris-versicolor", "Iris-virginica"])

disp.plot()


"naive bayes"

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X,Y)
print(gnb.score(X,Y))
y_NB=gnb.predict(X)

connb=confusion_matrix(Y,y_NB)
disp=ConfusionMatrixDisplay(confusion_matrix=connb, display_labels=["iris-setosa", "Iris-versicolor", "Iris-virginica"])
disp.plot()


