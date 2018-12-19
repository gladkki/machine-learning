# предсказание победы в dota2 при помощи градиентного бустинга
# и логистической регресии

import pandas
import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

features_train = pandas.read_csv('features.csv', index_col='match_id')
target = features_train['radiant_win']  # целевая переменная

# удалим признаки, которые заглядывают в будущее
features_train = features_train.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                                      'barracks_status_radiant', 'barracks_status_dire'], axis=1)

features_train = features_train.fillna(0)  # заменяем пропуски на нули

trees = [10, 20, 30, 40, 50, 100, 200]  # количество деревьев градиетного бустинга
quality = 0  # качество на кросс-валидации
for n in trees:
    #  реализация кросс-валидации для градиентого бустинга
    #  для ускорения работы программы этот цикл стоит комментить при работе с лог.регр.
    start_time = datetime.datetime.now()
    kf = KFold(n_splits=5, shuffle=True, random_state=241)
    gbc = GradientBoostingClassifier(n_estimators=n, random_state=241)
    score = cross_val_score(gbc, features_train, y=target, scoring='roc_auc', cv=kf)
    for i in range(0, 5):
        quality += score[i]
    quality = quality / 5
    print('качество на', n, 'деревьях', 'равно', quality)
    print('Time elapsed:', datetime.datetime.now() - start_time)
    quality = 0
