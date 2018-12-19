# предсказание победы в dota2 при помощи градиентного бустинга
# и логистической регресии

import pandas
import numpy
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

features_train = pandas.read_csv('features.csv', index_col='match_id')
features_test = pandas.read_csv('features_test.csv', index_col='match_id')
target = features_train['radiant_win']  # целевая переменная


def del_future(data):
    # удаляем признаки, заглядывающие в будущее
    data = data.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                      'barracks_status_radiant', 'barracks_status_dire'], axis=1)
    return data


# заменим пропуски на нули
features_train = features_train.fillna(0)
features_test = features_test.fillna(0)


def grad_bust(data_train, y):
    # кросс-валидация для градиетного бустинга
    quality = 0
    trees = [10, 20, 30, 40, 50, 100, 200]
    start_time = datetime.datetime.now()
    for n in trees:
        kf = KFold(n_splits=5, shuffle=True, random_state=241)
        gbc = GradientBoostingClassifier(n_estimators=n, random_state=241)
        score = cross_val_score(gbc, data_train, y=y, scoring='roc_auc', cv=kf)
        for i in range(0, 5):
            quality += score[i]
        quality = quality / 5
        print('качество на', n, 'деревьях', 'равно', quality)
        print('Time elapsed:', datetime.datetime.now() - start_time)
        quality = 0


print('количество идентификаторв персонажей =', max(features_train['r1_hero']))


def bag_of_words(data):
    # мешок слов. создаем 112 принаков, которые принимают значение 1
    # если игрок выбран стороной radiant, -1 - dire
    # и 0, если не выбран вовсе.
    # и 0, если не выбран вовсе
    x_pick = numpy.zeros((data.shape[0], 112))
    for i, match_id in enumerate(data.index):
        for p in range(5):
            x_pick[i, data.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            x_pick[i, data.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    x_pick = pandas.DataFrame(x_pick, index=data.index)
    return pandas.concat([data, x_pick], axis=1)


def del_cat_features(data):
    # удаляем категориальные признаки
    data = data.drop(['lobby_type'], axis=1)
    for i in range(1, 6):
        data = data.drop(['r' + str(i) + '_hero'], axis=1)
        data = data.drop(['d' + str(i) + '_hero'], axis=1)

    return data


def c_log_reg(data, y):
    # реализуем кросс-валидацию для логистической регрессии
    # и ищем оптимальный параметр С
    prm_c = numpy.linspace(0.01, 10, num=5)  # по этим С попробуем найти оптимальное качество
    quality = 0
    for c in prm_c:
        #  реализация кросс-валидации для логистической регрессии
        start_time = datetime.datetime.now()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        logreg = LogisticRegression(penalty='l2', C=c, random_state=42)
        score = cross_val_score(logreg, data, y=y, scoring='roc_auc', cv=kf)
        for i in range(0, 5):
            quality += score[i]
        quality = quality / 5
        print('качество с парметром С =', c, 'равно', quality)
        print('Time elapsed:', datetime.datetime.now() - start_time)
        quality = 0


def log_reg(data_train, data_test, y):
    # реализуем логистическую регрессию для оптимального С
    scaler = StandardScaler()  # для нормировки признаков
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    logreg = LogisticRegression(penalty='l2', C=0.11, random_state=42)
    logreg.fit(data_train, y)
    target_test = logreg.predict_proba(data_test)[:, 1]
    print('\nоценкки вероятностей равны\n', sorted(target_test))


# grad_bust(del_future(features_train), target)  запуск град.бустинга

features_train = del_future(features_train)

features_train = bag_of_words(features_train)
features_test = bag_of_words(features_test)

features_train = del_cat_features(features_train)
features_test = del_cat_features(features_test)

log_reg(features_train, features_test, target)

