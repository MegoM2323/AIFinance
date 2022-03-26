#https://ru.investing.com/crypto/bitcoin/historical-data
# From this site
#https://ru.tradingview.com/symbols/CRYPTOCAP-BTC.D/
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

PredictionNumber = 1# вход данные

PredictPeriod = 10

DayAmount = 3
#dfBTC = pd.read_csv("DataBase/BitcoinInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/ETHInvesting.csv") #Хорошо 5000 5100 5200
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/BitcoinInvesting.csv") # 67000
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/BCH.csv") #
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/BNBInvesting.csv") #Хорошо 660 n
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/LINKInvesting.csv") # So intersting 35 36
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/DASHInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/CHZ.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/DOTInvesting.csv")
df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/LTC.csv") # 215 will be
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/QTUMInvesting.csv") #Хорошо
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/XRPInvesting.csv") # докупи скоро по 1.2 1.15
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/SOLInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/BNT.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/ONT.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/AVAXInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/VETInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/TRX.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/CAKEInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/ADAInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/SUSHIInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/OMGInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/UNIInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/ETCInvesting.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/XLM.csv")
#df = pd.read_csv("C:\\Users\qwert\PycharmProjects\MainPythonProject\AIFinance\DataBase/XMR.csv")

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

# Очишение баз данных
CleaininDF(df)
joined = df

#Выбор периода отслеживания

ResPredictions = [[], [], []]
#Ввод информации
massive_input = []
for i in range(DayAmount + 1):
    massive_input.append(joined['Цена'].iloc[PredictionNumber+i])
massive_input = pd.Series(massive_input, dtype='float64')
#print(massive_input)

for i in range(PredictPeriod):
    # Сама модель
    x_test = joined['Цена'][PredictPeriod: -DayAmount]
    print(x_test)
    for i in range(DayAmount):
        x_test = pd.concat([x_test, joined['Цена'][PredictPeriod + i: -(DayAmount-i)]], axis=1)
        print(x_test)

    print(x_test)
# 90 % work
'''
Через листы неособо заработало
         10       11       12       13    ...   1900   1901   1902   1903
Цена  191.743  189.552  196.498  190.319  ...  3.720  3.780    NaN    NaN
Цена      NaN  189.552  196.498  190.319  ...  3.720  3.780  3.810    NaN
Цена      NaN      NaN  196.498  190.319  ...  3.720  3.780  3.810  3.810
'''

y_test = joined["Цена"][:-DayAmount-PredictPeriod]
# 10 % work
lr = LinearRegression()  # создание модели
lr.fit(x_test, y_test)
predictions_y_t = lr.predict(x_test)
test_deviation = np.sqrt(mean_squared_error(y_test, predictions_y_t))

ResPredictions[0].append(lr.predict(massive_input).reshape(-1, 1))

if ResPredictions[0][0] <= ResPredictions[0][-1]:
    plt.plot(range(len(ResPredictions[0])), ResPredictions[0], linestyle='-', marker='o', linewidth=2, color='green', label='Цена')
else:
    plt.plot(range(len(ResPredictions[0])), ResPredictions[0], linestyle='-', marker='o', linewidth=2, color='red', label='Цена')
plt.show()

# Ввывод информации
