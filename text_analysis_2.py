# coding=utf-8
import numpy as np
import time
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print "Elapsed time: {:.3f} sec".format(time.time() - self._startTime)

# subset = 'train', 'test' or 'all'
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
feature = newsgroups.data
target = newsgroups.target

# числовое представление текста
tf= TfidfVectorizer()
feature = tf.fit_transform(feature)

parameter_C = []
for i in range(-5, 6):
    parameter_C.append(10 ** i)

#ищем оптимальный параметер С с помощью кросс-валидации для метода опорных векторов
c_max = 0
sum = 0
max = 0
with Profiler() as p:
    for c in parameter_C:
        kf = KFold(n_splits=5, shuffle=True, random_state=241)
        svc = SVC(C=c, kernel='linear', random_state=241)
        score = cross_val_score(svc, feature, y=target, scoring='accuracy', cv=kf)
        for i in range(0, 5):
            sum += score[i]
        mean = sum / 5
        if mean > max:
            max = mean
            c_max = c
        sum = 0

svc = SVC(C=1, kernel='linear', random_state=241)
svc.fit(feature, target)

# to.array выводит массив с теми же элементами, что и у матрицы svc.coef_
most_important_words_indexes = np.argsort(abs(svc.coef_.toarray()[0]))[-10:]
most_important_words = np.array(tf.get_feature_names())[most_important_words_indexes]
most_important_words_sorted = sorted(most_important_words)

# join превращает массив в единую строку, расстояние между элементами строки " "
resulting_string = " ".join(most_important_words_sorted)

file_answer = open("answer_2.txt", "w")
file_answer.write(resulting_string)
file_answer.close()

print svc.coef_.toarray()
print type(svc.coef_)
