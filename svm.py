# coding=utf-8
import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('svm-data.csv', header=None)
target = data[0]
feature = data.drop([0], axis='columns')

# метод опорных векторов
svc = SVC(C=100000, kernel='linear', random_state=241)
svc.fit(feature, target)
support = svc.support_

print support