import pylab as plt
import math
import pandas as pd
import numpy as np
from skimage.io import imread, imsave
from skimage import img_as_float
from sklearn.cluster import KMeans

image = imread('parrots.jpg')
# вывод на экран
# plt.imshow(image, cmap='hot')
# plt.show()

image = img_as_float(image)
n, m, k = image.shape  # n высота, m ширина, k R G B
re_image = np.reshape(image, (n * m, k))  # n * m количество пикселей
data = pd.DataFrame(re_image, columns=['R', 'G', 'B'])  # матрица объект-признаки


def psnr(a, b):
    mean = np.mean((a - b) ** 2)
    return 10 * math.log10(1 / mean)


def cluster(data, i):
    # метод к средних
    data = data.copy()
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=241)
    data['clusters'] = kmeans.fit_predict(data)

    # для каждого кластера находим медиану по столбцам R, G, B
    median = data.groupby('clusters').median().values
    data_median = [median[element] for element in data['clusters'].values]

    # для каждого кластера находим среднее по столбцам R, G, B
    means = data.groupby('clusters').mean().values
    data_means = [means[element] for element in data['clusters'].values]
    return data_median, data_means


for i in range(1, 21):
    data_median, data_means = cluster(data, i)
    print(psnr(re_image, data_median), psnr(re_image, data_means))
    if psnr(re_image, data_median) > 20 or psnr(re_image, data_means) > 20:
        print(i)
        break




