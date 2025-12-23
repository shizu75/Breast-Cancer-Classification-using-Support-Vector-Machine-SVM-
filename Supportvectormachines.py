import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\Internship\cell_samples.csv")
mag_data = data[data['Class']== 4][0:200]
beni_data = data[data['Class']== 2][0:200]

axes = beni_data.plot(kind = 'scatter',  x = 'Clump', y = 'UnifSize', color = 'r', label = 'Benign')
mag_data.plot(kind = 'scatter',  x = 'Clump', y = 'UnifSize', color = 'b', label = 'Malignant', ax = axes)
plt.show()

data = data[pd.to_numeric(data['BareNuc'], errors = 'coerce').notnull()]
data['BareNuc']  = data['BareNuc'].astype('int')

feature = data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature)

Y = np.array(data['Class'])

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm
SV = svm.SVC()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state =  0)
SV.fit(X_train, Y_train)
prediction = SV.predict(X_test)
print(SV.support_vectors_)
print(confusion_matrix(Y_test, prediction))
print(accuracy_score(Y_test, prediction))
