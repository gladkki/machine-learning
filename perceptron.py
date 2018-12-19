# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data_train = pd.read_csv('perceptron-train.csv', header = None)
data_test = pd.read_csv('perceptron-test.csv', header = None)

# целевая переменная обучающей выборки
target_train = data_train[0]

# признаки обучающей выборки
feature_train = data_train.drop([0], axis='columns')

# начинаем обучать с помощью перцептрона
clf = Perceptron(random_state=241)
clf.fit(feature_train, target_train)

target_test = data_test[0]
feature_test = data_test.drop([0], axis='columns')

# ответы алгоритма
predictions = clf.predict(feature_test)

# считаем долю правильных ответов
score = accuracy_score(target_test, predictions)

# теперь проделаем все то же, но с нормализацией
scaler = StandardScaler()
feature_train_scaled = scaler.fit_transform(feature_train)
feature_test_scaled = scaler.transform(feature_test)
clf_scaled = Perceptron(random_state=241)
clf_scaled.fit(feature_train_scaled, target_train)
predictions_scaled = clf_scaled.predict(feature_test_scaled)
score_scaled = accuracy_score(target_test, predictions_scaled)

delta = score - score_scaled
print score, score_scaled, delta