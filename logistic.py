# coding=utf-8
import pandas as pd
import math
from sklearn.metrics import roc_auc_score

data = pd.read_csv('data-logistic.csv', header=None)
target = data[0]
feature = data.drop([0], axis=1)

def w_1(w1, w2, k, c, feature, target):
    # находим первый весовой коэффициент
    sum_1 = 0
    length = len(feature)
    for i in range(0, length):
        exp = 1 + math.exp(-target[i] * (w1 * feature[1][i] +  w2 * feature[2][i]))
        sum_1 += target[i] * feature[1][i] * (1 - 1 / exp)
    return w1 + (k / length) * sum_1 - k * c * w1

def w_2(w1, w2, k, c, feature, target):
    # находим второй весовой коэффициент
    sum_2 = 0
    length = len(feature)
    for i in range(0, length):
        exp = 1 + math.exp(-target[i] * (w1 * feature[1][i] + w2 * feature[2][i]))
        sum_2 += target[i] * feature[2][i] * (1 - 1 / exp)
    return w2 + (k / length) * sum_2 - k * c * w2


def grad(feature, target, w1 = 0, w2 = 0, k = 0.1, c = 0):
    # реализовываем метод градиентного спуска
    i = 0
    while True:
        w_1_new = w_1(w1, w2, k, c, feature, target)
        w_2_new = w_2(w1, w2, k, c, feature, target)
        if math.sqrt((w_1_new - w1) ** 2 + (w_2_new - w2) ** 2) <= 1e-5 or i > 10000:
            break
        else:
            w1 = w_1_new
            w2 = w_2_new
        i += 1

    w = [w1, w2]
    return w

print grad(feature, target)
w1 = grad(feature, target, c=0)[0]
w2 = grad(feature, target, c=0)[1]

def a(feature, w1, w2, i):
    return 1 / math.exp(-w1 * feature[1][i] - w2 * feature[2][i])

score = []
for i in range(0, len(feature)):
    score.append(a(feature, w1, w2, i=i))

# score = feature.apply(lambda x: a(x, w1, w2), axis=1)
roc = roc_auc_score(target, score)

print roc


