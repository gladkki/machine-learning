# coding=utf-8
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

boston = load_boston()
# file1 = open("feature.txt", "w")
# file1.write(repr(boston.data))
# file1.close()
# file2 = open("target.txt", "w")
# file2.write(repr(boston.target))
# file2.close()

feature = pd.DataFrame(boston.data)
target = pd.DataFrame(boston.target)

# привел признаки к одному масштабу
feature = scale(feature)

# разбивает отрезок от 1 до 10 на 200 частей
parameter = np.linspace(1, 10, num = 200)
sum = 0
max = -1000
pm_max = 0

for pm in parameter:
    neighbor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=pm, metric='minkowski')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(neighbor, feature, y = target, scoring='neg_mean_squared_error', cv=kf)
    for i in range(0,5):
        sum += score[i]
    mean = sum / 5
    if mean > max:
        max = mean
        pm_max = pm
    sum = 0

print max, pm_max