# coding=utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data = pd.read_csv('salary-train.csv')
test = pd.read_csv('salary-test-mini.csv')
fd = data['FullDescription'].str.lower()
ln = data['LocationNormalized'].str.lower()
ct = data['ContractTime']
sn = data['SalaryNormalized']

# заменяем все, кроме букв и цифр, на пробелы
fd = fd.replace('[^a-zA-Z0-9]', ' ', regex = True)

# заменяем слова в показатель tf-idf
tf = TfidfVectorizer()
fd = tf.fit_transform(fd)

# заменяем пропуски Nan а строку 'nan'
ln.fillna('nan', inplace=True)
ct.fillna('nan', inplace=True)

# one hot кодирование
dv = DictVectorizer()
new_data = pd.DataFrame({'LocationNormalized': ln, 'ContractTime': ct})
train = dv.fit_transform(new_data[['LocationNormalized', 'ContractTime']].to_dict('records'))

data_train = hstack([fd, train])

ridge = Ridge(alpha=1, random_state=241)
ridge.fit(data_train, sn)

fd_test = test['FullDescription'].str.lower()
ln_test = test['LocationNormalized'].str.lower()
ct_test = test['ContractTime']
sn_test = test['SalaryNormalized']

fd_test = fd_test.replace('[^a-zA-Z0-9]', ' ', regex = True)
fd_test = tf.transform(fd_test)

ln_test.fillna('nan', inplace=True)
ct_test.fillna('nan', inplace=True)

new_data_test = pd.DataFrame({'LocationNormalized': ln_test, 'ContractTime': ct_test})
train_test = dv.transform(new_data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

data_train_test = hstack([fd_test, train_test])

predict = ridge.predict(data_train_test)
print (predict)
