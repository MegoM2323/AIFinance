import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("DataBase/BitcoinInvesting.csv")
'''print(df.head(10))
print(df.columns)
n = df.columns
print(df.shape)
print(df.info())'''
for i in range(len(df['Цена'])):
    '''c1 = df['Цена'][i][:-2].split('.')
    c1 = df['Цена'][i][:-2].split('.')
    c1 = df['Цена'][i][:-2].split('.')
    if len(c1) > 1:
        df['Цена'][i] = int(c1[0] + c1[1])
    else:
        df['Цена'][i] = int(c1[0])
    print(df['Цена'][i])'''

    c1 = df['Цена'][i][:-2].split(".")
    df['Цена'][i] = "".join(c1)

    c2 = df['Откр.'][i][:-2].split(".")
    df['Откр.'][i] = int("".join(c2))

    c3 = df['Макс.'][i][:-2].split(".")
    df['Макс.'][i] = int("".join(c3))

    c4 = df['Мин.'][i][:-2].split(".")
    df['Мин.'][i] = int("".join(c4))

    c5 = df['Объём'][i][:-1].split(",")
    df['Объём'][i] = int("".join(c5) + "0")

    c6 = df['Изм. %'][i][:-1].split(",") # Проценты остаются без изменений т.к. они полностью рабочие и стабильные
    df['Изм. %'][i] = float(".".join(c6))

lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(df[0:-1].drop("Дата", axis=1), df['Цена'][1:], test_size = 0.3)
lr.fit(x_train, y_train)
predictions_y = lr.predict(x_train)
train_deviation = np.sqrt(mean_squared_error(y_train, predictions_y))
predictions_y_t = lr.predict(x_test)
test_deviation = np.sqrt(mean_squared_error(y_test, predictions_y_t))

k = [61312,60693,61728,59738,37050,1.03] #60.866,5
k1 = [61309,61842,62393,60005,50060,-0.86] #60.915
k2 = [60915,61310,62430,59612,61210,-0.64] #63.261
isp = np.array(k)
'''sum = 0
count = 0
for i in range(10):
    sum += lr.predict(isp.reshape(1, -1))
    count += 1'''
#p1 = float(lr.predict(isp.reshape(1, -1)))
#kd = [p1,61310,62430,59612,61210,-0.64]
#isp2 = np.array(kd)
#print(p1)
#print(float(lr.predict(isp2.reshape(1, -1))))
print(float(lr.predict(isp.reshape(1, -1))))
print(train_deviation, test_deviation)