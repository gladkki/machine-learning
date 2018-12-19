# coding=utf-8
# реализовываем метод главных компонент

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv('close_prices.csv')
data = data.drop(['date'], axis=1)
print data[[1]]
index = pd.read_csv('djia_index.csv')
index = index.drop(['date'], axis=1)
dji = index['^DJI']
pca = PCA(n_components=10)
pca.fit(data)

# массив содержит проценты долей дисперсий, сохранившихся после pca
dispersion = pca.explained_variance_ratio_

# number_comp = 0
# sum = 0
# for i in range(0, len(dispersion)):
#     sum += dispersion[i]
#     number_comp += 1
#     if sum >= 0.9:
#         break

fit_data = pca.transform(data)
fit_data = pd.DataFrame(fit_data)
fit_data_0 = fit_data[0]

# тонкий момент: index не прокатил, пришлось использовать dji
pearson = np.corrcoef(fit_data_0, dji)

components = pca.components_
components = pd.DataFrame(components)

print np.argsort(abs(components.iloc[0]))


