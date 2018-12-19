import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from IPython.display import Image
import pydotplus
import graphviz


data = pd.read_csv('titanic.csv', index_col='PassengerId')
sex = data['Sex']
sex = sex.replace(to_replace=['male', 'female'], value=[1, 0])  # replace value in series
new_data = pd.DataFrame(data=data, columns=['Survived','Pclass', 'Age', 'Sex', 'Fare'])
new_data = new_data.replace(to_replace=['male', 'female'], value=[1, 0])
new_data = new_data.dropna()  # delete object with nan
finish_data = new_data[['Pclass', 'Age', 'Sex', 'Fare']]
clf = DecisionTreeClassifier(random_state=241)
y = new_data['Survived']  # target variable
X = finish_data
clf.fit(X, y)
export_graphviz(clf,  out_file='tree.dot', filled=True)
importance = clf.feature_importances_
print  finish_data, importance
