# предсказанние победы в dota2 при помощи логистической регрессии

import pandas
import numpy
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

features_train = pandas.read_csv('features.csv', index_col='match_id')
features_test = pandas.read_csv('features_test.csv', index_col='match_id')
target = features_train['radiant_win']  # целевая переменная

# удалим признаки, которые заглядывают в будущее
features_train = features_train.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                                      'barracks_status_radiant', 'barracks_status_dire'], axis=1)

features_train = features_train.fillna(0)  # заменяем пропуски на нули
features_test = features_test.fillna(0)

# мешок слов. создаем 112 принаков, которые принимают значение 1
# если игрок выбран стороной radiant, -1 - dire
# и 0, если не выбран вовсе
X_pick = numpy.zeros((features_train.shape[0], 112))
for i, match_id in enumerate(features_train.index):
    for p in range(5):
        X_pick[i, features_train.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, features_train.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

X_pick = pandas.DataFrame(X_pick, index=features_train.index)
features_train = pandas.concat([features_train, X_pick], axis=1)

# тоже самое, но для тестовой выборки
X_pick2 = numpy.zeros((features_test.shape[0], 112))
for i, match_id in enumerate(features_test.index):
    for p in range(5):
        X_pick2[i, features_test.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick2[i, features_test.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
X_pick2 = pandas.DataFrame(X_pick2, index=features_test.index)
features_test = pandas.concat([features_test, X_pick2], axis=1)

print(max(features_train['r1_hero']))  # количество идентификатор персонажей

# удалим категориальные признаки и узнаем количество уникальных героев
features_train = features_train.drop(['lobby_type'], axis=1)
features_test = features_test.drop(['lobby_type'], axis=1)
for i in range(1, 6):
    features_train = features_train.drop(['r' + str(i) + '_hero'], axis=1)
    features_train = features_train.drop(['d' + str(i) + '_hero'], axis=1)
    features_test = features_test.drop(['r' + str(i) + '_hero'], axis=1)
    features_test = features_test.drop(['d' + str(i) + '_hero'], axis=1)

# нормируем признаки
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

prm_C = numpy.linspace(0.01, 10, num=5)  # по этим С попробуем найти оптимальное качество
quality = 0
for c in prm_C:
    #  реализация кросс-валидации для логистической регрессии
    start_time = datetime.datetime.now()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    log_reg = LogisticRegression(penalty='l2', C=c, random_state=42)
    score = cross_val_score(log_reg, features_train, y=target, scoring='roc_auc', cv=kf)
    for i in range(0, 5):
        quality += score[i]
    quality = quality / 5
    print('качество с парметром С =', c, 'равно', quality)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    quality = 0

log_reg = LogisticRegression(penalty='l2', C=0.11, random_state=42)
log_reg.fit(features_train, target)
target_test = log_reg.predict_proba(features_test)[:, 1]
print(sorted(target_test))
target_test = pandas.Series(target_test)
print(target_test.mean())
