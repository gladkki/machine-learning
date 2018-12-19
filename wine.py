import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


name = []
max = 0
k_max = 0
sum = 0
name_2 = []

for i in range(0,14):
    name.append(i)
for i in range(1,14):
    name_2.append(i)

data_1 = pd.read_csv('wine.data', names = name)
target_name = data_1[0]
data_1 = scale(data_1)
data = pd.DataFrame(data_1)
data = data.rename(columns={0: 'huy'})
data = data.drop(['huy'], axis='columns')
print data
for k in range(1, 51):
    neighbors = KNeighborsClassifier(n_neighbors = k)
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    score = cross_val_score(neighbors,data, y = target_name, scoring = 'accuracy', cv=kf)
    for i in range(0,5):
        sum += score[i]
    mean = sum / 5
    if mean > max:
        max = mean
        k_max = k
    sum = 0
print max, k_max


