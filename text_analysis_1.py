# coding=utf-8
import time
import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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

with Profiler() as p:
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    svc = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(svc, grid, scoring='accuracy', cv=cv)
    gs.fit(feature, target)

c_max = gs.best_estimator_.C
svc = SVC(C=c_max, kernel='linear', random_state=241)
svc.fit(feature, target)

most_important_words_indexes = np.argsort(abs(svc.coef_.toarray()[0]))[-10:]
most_important_words = np.array(tf.get_feature_names())[most_important_words_indexes]
most_important_words_sorted = sorted(most_important_words)
resulting_string = " ".join(most_important_words_sorted)

file_answer = open("answer.txt", "w")
file_answer.write(resulting_string)
file_answer.close()

