import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

data = pd.read_csv('classification.csv')
score = pd.read_csv('scores.csv')

# true = data['true']
# pred = data['pred']
#
# tp = 0
# fp = 0
# tn = 0
# fn = 0
# for i in range(0,len(data)):
#     if true[i] == 1 and pred[i] == 1:
#         tp += 1
#     elif true[i] == 1 and pred[i] == 0:
#         fn += 1
#     elif true[i] == 0 and pred[i] == 1:
#         fp += 1
#     elif true[i] == 0 and pred[i] == 0:
#         tn += 1
#
# accuracy = accuracy_score(true, pred)
# precision = precision_score(true, pred)
# recall = recall_score(true, pred)
# f1 = f1_score(true, pred)

# print accuracy, precision, recall, f1

true_score = score['true']
logreg = score['score_logreg']
svm = score['score_svm']
knn = score['score_knn']
tree = score['score_tree']

# roc1 = roc_auc_score(true_score, logreg)
# roc2 = roc_auc_score(true_score, svm)
# roc3 = roc_auc_score(true_score, knn)
# roc4 = roc_auc_score(true_score, tree)

prc = precision_recall_curve

prc1, rec1, thr1 = prc(true_score, logreg)
prc2, rec2, thr2 = prc(true_score, svm)
prc3, rec3, thr3 = prc(true_score, knn)
prc4, rec4, thr4 = prc(true_score, tree)

score1 = pd.DataFrame({'prc1': prc1, 'rec1': rec1})
score2 = pd.DataFrame({'prc2': prc2, 'rec2': rec2})
score3 = pd.DataFrame({'prc3': prc3, 'rec3': rec3})
score4 = pd.DataFrame({'prc4': prc4, 'rec4': rec4})

print score1[score1['rec1'] > 0.7]['prc1'].max()
print score2[score2['rec2'] > 0.7]['prc2'].max()
print score3[score3['rec3'] > 0.7]['prc3'].max()
print score4[score4['rec4'] > 0.7]['prc4'].max()

