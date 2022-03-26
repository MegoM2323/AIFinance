# Добавленно распознование обьема (Исправлен баг) Теперь 1миллион и 1 тясяча это не одно и тоже(Обьём)
#https://ru.investing.com/crypto/bitcoin/historical-data
# From this site
#https://ru.tradingview.com/symbols/CRYPTOCAP-BTC.D/
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dfBTC = pd.read_csv("../DataBase/BTC.csv")
df = pd.read_csv("../DataBase/BNB.csv")
#df = pd.read_csv("LINKInvesting.csv")
#df = pd.read_csv("ETHInvesting.csv")
#df = pd.read_csv("DOTInvesting.csv")
#print(df.head(10))

#------------------------------------------------------------------------------------------------------------------
'''
print(df.columns)
n = df.columns
print(df.shape)
print(df.info())'''
#------------------------------------------------------------------------------------------------------------------

def InputRefresh(massive):
    i = 0
    while i < len(massive):
        if massive[i] == "'" or massive[i] == " " or massive[i] == '"' or massive[i] == '' or massive[i] == ',':
            massive.remove(massive[i])
        i += 1
    for i in range(1, 5):
        massive[i] = '.'.join("".join(massive[i].split('.')).split(","))

    word = massive[5][-1]
    if word == "М": #Обработка при миллионном обороте
        c5 = massive[5][:-1].split(",")
        massive[5] = int("".join(c5) + "0000")
    else: # Обработка при тысечном обороте
        c5 = massive[5][:-1].split(",")
        massive[5] = int("".join(c5) + "0")

    c6 = massive[6][:-1].split(",")
    massive[6] = float(".".join(c6))
    return massive[1:]

def CleaininDF(df):
    for i in range(len(df['Цена'])):
        df['Цена'][i] = '.'.join("".join(df['Цена'][i].split('.')).split(","))  # Обработка цены остаётся без изменений т.к. она полностью рабочая и стабильная
        # df['Цена'][i] = "".join(c1)

        df['Откр.'][i] = '.'.join("".join(df['Откр.'][i].split('.')).split(","))
        # df['Откр.'][i] = int("".join(c2))

        df['Макс.'][i] = '.'.join("".join(df['Макс.'][i].split('.')).split(","))
        # df['Макс.'][i] = int("".join(c3))

        df['Мин.'][i] = '.'.join("".join(df['Мин.'][i].split('.')).split(","))
        # df['Мин.'][i] = int("".join(c4))

        word = df['Объём'][i][-1]
        if word == "М": #Обработка при миллионном обороте
            c5 = df['Объём'][i][:-1].split(",")  # Тоже работает стабильно
            df['Объём'][i] = int("".join(c5) + "0000")
        else: # Обработка при тысечном обороте
            c5 = df['Объём'][i][:-1].split(",")  # Тоже работает стабильно
            df['Объём'][i] = int("".join(c5) + "0")

        c6 = df['Изм. %'][i][:-1].split(",")  # Обработка процентов остаются без изменений т.к. они полностью рабочие и стабильные
        df['Изм. %'][i] = float(".".join(c6))


def GenerateReriodProcent(massive, period, name): # потом при стабтльной работе добавим 2 аргумент
    l0, l, l1 = [], [], []
    for i in range(period + 1, len(massive)):
        l0.append(i)
        l.append(massive[i])
        l1.append(round((float(massive[i])/float(massive[i-period]) - 1) * 100, 2))
    for i in range(period+1):
        l0.append(i)
        l.append(massive[i])
        l1.append(0)
    l0 = np.array(l0)
    l = np.array(l)
    l1 = np.array(l1)
    res = pd.DataFrame(l1)
    res.columns = [name + '%']
    return res

# Очишение баз данных
CleaininDF(df)
CleaininDF(dfBTC)

timeBTC_massive = GenerateReriodProcent(dfBTC['Цена'], 30, 'Мес')
timeBTC_massive2 = GenerateReriodProcent(dfBTC['Цена'], 7, 'Неделя')
joinedBTC = pd.concat([dfBTC, timeBTC_massive, timeBTC_massive2], axis=1)#df.merge(time_massive, on='Цена', how='lef')

time_massive = GenerateReriodProcent(df['Цена'], 30, 'Мес')
time_massive2 = GenerateReriodProcent(df['Цена'], 7, 'Неделя')
joined = pd.concat([df, time_massive, time_massive2], axis=1)#df.merge(time_massive, on='Цена', how='lef')

Alljoined = joined.merge(joinedBTC, on='Дата', how='left')

#------------------------------------------------------------------------------------------------------------------
#print(Alljoined.columns)
#print(Alljoined.shape)
#Alljoined = Alljoined[:-40]
#print(joined.head(15))
#print(joinedBTC.head(15))
#------------------------------------------------------------------------------------------------------------------

#Выбор периода отслеживания
PredictPeriod = int(input("Period:\n"))
# Сама модель
x_train, x_test, y_train, y_test = train_test_split(Alljoined.drop("Дата", axis=1)[PredictPeriod:], Alljoined['Цена_x'][:-PredictPeriod], test_size=0.3)
lr = LinearRegression() #создание модели
lr.fit(x_train, y_train) # тренировка
lr.fit(x_test, y_test)
predictions_y = lr.predict(x_train)
train_deviation = np.sqrt(mean_squared_error(y_train, predictions_y))
predictions_y_t = lr.predict(x_test)
test_deviation = np.sqrt(mean_squared_error(y_test, predictions_y_t))

#Ввод информации
massive_input = np.array(InputRefresh(list(input("Reading for First Monet:\n").split('"'))))
massive_input = np.append(massive_input, [float(input()), float(input())]) # 'Месяц:\n' 'Неделя:\n'

#print(massive_input)
#print(*lr.predict(massive_input.reshape(1, -1)))

massive_input_BTC = np.array(InputRefresh(list(input("Reading BTC:\n").split('"'))))
massive_input_BTC = np.append(massive_input_BTC, [float(input()), float(input())]) # 'Месяц:\n' 'Неделя:\n'

All_input = np.append(massive_input, massive_input_BTC) # Совмешение вводов

# Ввывод информации
print(*lr.predict(All_input.reshape(1, -1)))
print(train_deviation, test_deviation) # Ввывод отклонений на тренировочном и на тестовом датасете