# Прогноз цен на автомобили
# Множественная линейная регрессия
# Python -v 3.11.4 x64. Pycharm -v 2023.1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

import os
for dirname, _, filenames in os.walk('Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv(r'Data\CarPrice_Assignment.xls')
print(data.head())

data.info()
data.describe()
data.select_dtypes('object').describe()


data.select_dtypes('number').describe()

#Замена Всех чисел цифрами (One => 1)
print(data['cylindernumber'].unique())
data['cylindernumber'] = data['cylindernumber'].replace({'four':4, 'six':6, 'five':5, 'three':3, 'twelve':12, 'two':2, 'eight':8})

print(data['doornumber'].unique())
data['doornumber'] = data['doornumber'].replace({'four':4, 'two':2})

data = data.drop(['car_ID','CarName'], axis=1)
data.head()

categories = data.select_dtypes('object').columns
categories

rows,cols=3,3
it=1
fig = plt.figure()
fig.set_figwidth(15)
#fig.set_figheight()
for i in categories:
    plt.subplot(rows,cols,it)
    sns.countplot(x=data.loc[:,i])
    it+=1

plt.tight_layout()
plt.show()

y = data['price']
x = data.drop('price',axis=1)

x = pd.get_dummies(x)
x.head()

ss = StandardScaler()

x = pd.DataFrame(ss.fit_transform(x), columns=x.columns)
x.head()

x_tr, x_test, y_tr, y_test = train_test_split(x,y, test_size=0.3, random_state=47)

lr = LinearRegression()
lr.fit(x_tr, y_tr)

y_predict_test_lr = lr.predict(x_test)
y_predict_train_lr = lr.predict(x_tr)

# Прогноз на тестовых данных и на обучающих данных
plt.plot(y_test, y_predict_test_lr, 'r.')
plt.plot(y_tr, y_predict_train_lr, 'bx')
plt.legend(['Прогноз на тестовых данных', 'Прогноз на обучающих данных'])
plt.show()

print("Оценка тестовых данных по сравнению с тренировочными данными")
print(metrics.r2_score(y_test, y_predict_test_lr),metrics.r2_score(y_tr, y_predict_train_lr))

print("Коэффициент")
print(pd.DataFrame(lr.coef_,index=x_tr.columns,columns=['Coefficients']).transpose())