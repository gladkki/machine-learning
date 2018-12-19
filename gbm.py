import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('gbm-data.csv')
target = data['Activity']
feature = data.drop(['Activity'], axis=1)

target = target.values
feature = feature.values
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.8, random_state=241)

learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
loss_train = []
loss_test = []
min_loss = []

gbc = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=0.2)
gbc.fit(X_train, y=y_train)
score_train = gbc.staged_decision_function(X_train)
score_test = gbc.staged_decision_function(X_test)
for pred in score_train:
    loss_train.append(log_loss(y_train, [1 / (1 + math.exp((-1) * y_pred)) for y_pred in pred]))
for pred in score_test:
    loss_test.append(log_loss(y_test, [1 / (1 + math.exp((-1) * y_pred)) for y_pred in pred]))
    min_loss.append(min(loss_test))

min_value = min(min_loss)
min_index = min_loss.index(min_value)

rfc = RandomForestClassifier(n_estimators=300, random_state=241)
rfc.fit(X_train, y_train)
predict = rfc.predict_proba(X_test)
log_loss_rfc = log_loss(y_test, predict)

print log_loss_rfc

