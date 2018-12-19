import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


data = pd.read_csv('abalone.csv')
target = data['Rings']
feature = data.drop('Rings', axis=1)
feature['Sex'] = feature['Sex'].map(lambda x: 1 if x =='M' else(-1 if x == 'F' else 0))

sum = 0
n_forest = 0
n_min = 0
for n_forest in range(1, 51):
    clf = RandomForestRegressor(n_estimators=n_forest, random_state=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    score = cross_val_score(clf, feature, y=target, scoring='r2', cv=kf)
    for i in range(0,5):
        sum += score[i]
    mean = sum / 5
    print mean
    if mean > 0.52:
        n_min = n_forest
    sum = 0

print n_min