#https://ru.investing.com/crypto/bitcoin/historical-data
# From this site
#https://ru.tradingview.com/symbols/CRYPTOCAP-BTC.D/
import datetime as dt
import pandas as pd
import numpy as np
import BaseConfig
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
def GeneratePeriodProcent(massive, period, name): # потом при стабтльной работе добавим 2 аргумент
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
def AIModel(massive_input, joined, PredictPeriod):
    ResPredictions = [[], [], []]
    ResPredictions[0].append(float(massive_input[0])) # добавление изначальной цены для просмотра разницы
    print(massive_input[0])
    for i in range(PredictPeriod):
        # Сама модель
        x_train, x_test, y_train, y_test = train_test_split(joined.drop(["Дата", 'Объём', "Изм. %"], axis=1)[PredictPeriod:],
                                                            joined.drop(["Дата", 'Объём', "Изм. %"], axis=1)[:-PredictPeriod],
                                                            test_size=0.5)

        lr = LinearRegression()  # создание модели
        lr.fit(joined.drop(["Дата", 'Объём', "Изм. %"], axis=1)[PredictPeriod:], joined.drop(["Дата", 'Объём', "Изм. %"], axis=1)[:-PredictPeriod])  # тренировка
        a, b =joined.drop("Дата", axis=1)[PredictPeriod:], joined.drop(["Дата", 'Объём', "Изм. %"], axis=1)[:-PredictPeriod]
        print(a.head())
        #print(b.head())
        predictions_y = lr.predict(x_train)
        train_deviation = np.sqrt(mean_squared_error(y_train, predictions_y))
        predictions_y_t = lr.predict(x_test)
        test_deviation = np.sqrt(mean_squared_error(y_test, predictions_y_t))
        # Сохранение результатов в лист
        ResPredictions[0].append(lr.predict(massive_input.reshape(1, -1))[0][0])
        ResPredictions[1].append(lr.predict(massive_input.reshape(1, -1))[0][2])  # Max
        ResPredictions[2].append(lr.predict(massive_input.reshape(1, -1))[0][3])  # Min
    return ResPredictions
def ShowResult(ResPredictions, NameDB):
    fig, ax = plt.subplots(figsize=(1, 1))

    if ResPredictions[0][0] <= ResPredictions[0][-1]:
        ax.plot(range(len(ResPredictions[0])), ResPredictions[0], linestyle='-', marker='o', linewidth=2, color='green',
                label='Цена')
    else:
        ax.plot(range(len(ResPredictions[0])), ResPredictions[0], linestyle='-', marker='o', linewidth=2, color='red',
                label='Цена')
    ax.plot(range(len(ResPredictions[1])), ResPredictions[1], linestyle='--', marker='o', linewidth=1, color='Blue',
            label='Макс', alpha=0.15)
    ax.plot(range(len(ResPredictions[2])), ResPredictions[2], linestyle='--', marker='o', linewidth=3, color='yellow',
            label='Мин', alpha=1)
    ax.legend(loc='lower left')
    ax.set_title(NameDB[:-4], fontsize=16)
    plt.show()
    # plt.savefig('saved_graph.png')
    # print(ResPredictions)
    # Ввывод информации
def SaveResults(ResPredictions, CryptoName, PredictionNumber):
    StrindForSave = ''
    for i in range(len(ResPredictions)):
        if ResPredictions[i] < 150:
            StrindForSave += str(round(ResPredictions[i], 3)) + ' '
        else:
            StrindForSave += str(round(ResPredictions[i], 1)) + ' '
    with open('../SaveResults.txt', 'r') as f:
        l = f.readlines()
    with open('../SaveResults.txt', 'w') as f:
        StrindForSave = str(PredictionNumber) + ' ' + CryptoName[:CryptoName.index(".")] + ' : ' + StrindForSave + '\n'
        print(StrindForSave)
        f.write(StrindForSave)
        f.writelines(l)
'''
Сделать показ предыдущего графика
Сделать функционализацию +
Сделать повтор анализа и показ на графике
Сделать запись файлов в текстовый документ
Показ среднего арифметического 
'''

def MainPrediction(PredictionNumberOfDay = 0, PredictPeriod = 3, CryptoNames = BaseConfig.CryptoNames):
    #PredictionNumberOfDay  вход данные
    #PredictPeriod период до которого идут предсказания
    # CryptoNames = ['EGLD.csv']
    for NameDB in CryptoNames or NameDB not in CryptoNames:
        if NameDB in BaseConfig.NoUse:
            continue
        df = pd.read_csv('DataBase/' + NameDB)
        # Очишение баз данных
        CleaininDF(df)
        #Генерация Мес и Недельных процентов
        time_massive = GeneratePeriodProcent(df['Цена'], 30, 'Мес')
        time_massive2 = GeneratePeriodProcent(df['Цена'], 7, 'Неделя')
        #Обьединение в одну базу данных
        joined = pd.concat([df, time_massive, time_massive2], axis=1)
        #Ввод информации
        #massive_input = np.array(joined.iloc[PredictionNumberOfDay][1:])
        massive_input = np.array(df.drop(["Изм. %", 'Объём'], axis=1).iloc[PredictionNumberOfDay][1:])
        print(massive_input)
        print(df.head())
        #Получение прогнозов
        #ResPredictions = AIModel(massive_input, joined, PredictPeriod)
        ResPredictions = AIModel(massive_input, df, PredictPeriod)
        #Показ прогнозов
        #ShowResult(ResPredictions, NameDB)
        SaveResults(ResPredictions[0], NameDB, PredictionNumberOfDay)
    with open('../SaveResults.txt', 'r') as f:
        l = f.readlines()
    with open('../SaveResults.txt', 'w') as f:
        f.writelines(str(dt.datetime.now())[:-7] + '\n')
        f.writelines(l)
