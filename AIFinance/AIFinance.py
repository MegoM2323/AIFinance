#https://ru.investing.com/crypto/bitcoin/historical-data
# From this site
#https://ru.tradingview.com/symbols/CRYPTOCAP-BTC.D/
import datetime as dt
import pandas as pd
import numpy as np
#Warning: was commented
#C:\Users\qwert\PycharmProjects\MainPythonProject\venv\lib\site-packages\sklearn\base.py:445
import BaseConfig
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

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
        try:
            df['Цена'][i] = '.'.join("".join(df['Цена'][i].split('.')).split(","))  # Обработка цены остаётся без изменений т.к. она полностью рабочая и стабильная
            # df['Цена'][i] = "".join(c1)
        except:
            pass
        try:
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
        except:
            pass

def ClassSort(FilterFileName = 'SaveResults.txt'):
    with open(FilterFileName, 'r') as f:
        l = f.readlines()
    ErrorClass = []
    ExtraClass = []
    SuperExtraClass = []
    MidlClass = []
    LowClass = []
    for i in range(1, len(l[1:len(BaseConfig.CryptoNames)]) - len(BaseConfig.NoUse) + 2):
        try:
            lAnalis = l[i].split()
            if lAnalis[1] + '.csv' in BaseConfig.ErrorClass:
                ErrorClass.append(l[i])
            elif lAnalis[1] + '.csv' in BaseConfig.ExtraClass:
                ExtraClass.append(l[i])
            elif lAnalis[1] + '.csv' in BaseConfig.SuperExtraClass:
                SuperExtraClass.append(l[i])
            elif lAnalis[1] + '.csv' in BaseConfig.MidlClass:
                MidlClass.append(l[i])
            elif lAnalis[1] + '.csv' in BaseConfig.LowClass:
                LowClass.append(l[i])
        except:
            pass
    res = [SuperExtraClass, ExtraClass, MidlClass, LowClass, ErrorClass]
    return res
def SaveTime(SaveFileName = 'SaveResults.txt'):
    with open(SaveFileName, 'r') as f:
        l = f.readlines()
    with open(SaveFileName, 'w') as f:
        f.writelines(str(dt.datetime.now())[:-7] + '\n')
        f.writelines(l)
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
def MSEErrorRegion(price, deviation_1, deviation_2):
    if price < 1:
        deviation_1, deviation_2 = round(deviation_1, 4), round(deviation_2, 4)
    elif price < 50:
        deviation_1, deviation_2 = round(deviation_1, 2), round(deviation_2, 2)
    else:
        deviation_1, deviation_2 = int(deviation_1), int(deviation_2)
    return deviation_1, deviation_2

def AIModel(massive_input, joined, PredictPeriod):
    ResPredictions = [[], [], []]
    ResPredictions[0].append(float(massive_input[0])) # добавление изначальной цены для просмотра разницы
    print(massive_input[0])
    for i in range(PredictPeriod):
        # Сама модель
        x_train, x_test, y_train, y_test = train_test_split(joined.drop("Дата", axis=1)[PredictPeriod:],
                                                            joined.drop(["Дата", 'Объём'], axis=1)[:-PredictPeriod],
                                                            test_size=0.5)
        lr = LinearRegression()  # создание модели
        lr.fit(x_train, y_train)  # тренировка
        #lr.fit(x_test, y_test)
        predictions_y = lr.predict(x_train)
        train_deviation = np.sqrt(mean_squared_error(y_train, predictions_y))
        predictions_y_t = lr.predict(x_test)
        test_deviation = np.sqrt(mean_squared_error(y_test, predictions_y_t))
        # Сохранение результатов в лист
        ResPredictions[0].append(lr.predict(massive_input.reshape(1, -1))[0][0])
        ResPredictions[1].append(lr.predict(massive_input.reshape(1, -1))[0][2])  # Max
        ResPredictions[2].append(lr.predict(massive_input.reshape(1, -1))[0][3])  # Min
        ErrorMSE = MSEErrorRegion(ResPredictions[0][0], train_deviation, test_deviation)
    ResPredictions[0].append(ErrorMSE[0])
    ResPredictions[0].append(ErrorMSE[1])
    return ResPredictions
def GeneratePriceBefor(massive, period=30):
    res = pd.DataFrame(massive)
    for i in range(1, period + 1):
        try:
            res['Day_' + str(i)] = massive.shift(-i).fillna(massive)
        except Exception as E:
            print('Error with move(i) ' + i)
            pass
    return res[:-i]

def GeneratePriceBefor1(massive, period=30):
    res = pd.DataFrame(massive)
    for i in range(1, period + 1):
        res['Day_' + str(i)] = massive['Цена'].shift(-i).fillna(massive['Цена'])
        res['Volume_' + str(i)] = massive['Объём'].shift(-i).fillna(massive['Объём'])
    return res[:-i]
def AIModelPro(DataFrame, PredictPeriod, MonetName, PredictionNumberOfDay = 0):
    MonetName = MonetName[:MonetName.index('.')]
    try:
        PeriodDayBefor = BaseConfig.MonetErrorClassificator[MonetName]
    except:
        PeriodDayBefor = 60
    ResPredictions = []
    ResPredictions.append(float(DataFrame['Цена'].iloc[PredictionNumberOfDay])) # добавление изначальной цены для просмотра разницы
    DataFrame = GeneratePriceBefor(DataFrame['Цена'], PeriodDayBefor)
    for i in range(PredictPeriod):
        # Сама модель
        try:
            x_train, x_test, y_train, y_test = train_test_split(DataFrame[PredictPeriod:],
                                                                DataFrame["Цена"][:-PredictPeriod],
                                                                test_size=0.25)
            lr = LinearRegression()  # создание модели
            lr.fit(x_train, y_train)  # тренировка
            #lr.fit(x_test, y_test)
            predictions_y = lr.predict(x_train)
            train_deviation = np.sqrt(mean_squared_error(y_train, predictions_y))
            predictions_y_t = lr.predict(x_test)
            test_deviation = np.sqrt(mean_squared_error(y_test, predictions_y_t))
            #Входные данные
            massive_input = np.array(DataFrame.iloc[PredictionNumberOfDay])
            # Сохранение результатов в лист
            ResPredictions.append(*lr.predict(massive_input.reshape(1, -1))) # Может нужно будет исправление для лучшего предсказывания утром вот такое (iloc[0])
            #ResPredictions.append("                    ")
            ErrorMSE = MSEErrorRegion(ResPredictions[0], train_deviation, test_deviation)
            ResPredictions.append(ErrorMSE[0])
            ResPredictions.append(ErrorMSE[1])
        except Exception as E:
            print('Error', E)
    return ResPredictions
def ShowResult(ResPredictions, NameDB):
    fig, ax = plt.subplots(figsize=(1, 1))

    try:
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
    except:
        print('Error with ShowResult')
    ax.legend(loc='lower left')
    ax.set_title(NameDB[:-4], fontsize=16)
    plt.show()
    # plt.savefig('saved_graph.png')
    # print(ResPredictions)
    # Ввывод информации
def SortLastInfo():
    with open('SaveResults.txt', 'r') as f:
        l = f.readlines()
    res = []
    delta = []
    for i in range(1, len(l[1:len(BaseConfig.CryptoNames)]) - len(BaseConfig.NoUse) + 2):
        lAnalis = l[i].split()
        for j in range(3, 7):
            lAnalis[j] = float(lAnalis[j])
        if lAnalis[3] < lAnalis[4]:
            formula = round((lAnalis[4] / lAnalis[3] - 1) * 100, 2)
            if formula > 1.5:
                delta.append(formula)
                res.append(l[i])
    result = []
    for i in range(len(delta)):
        ind = delta.index(max(delta))
        delta.remove(max(delta))
        result.append(res.pop(ind))
    return result
def SaveResults(ResPredictions, CryptoName, PredictionNumber, SaveFileName = 'SaveResults.txt'):
    StrindForSave = ''
    for i in range(len(ResPredictions)):
        if ResPredictions[i] < 3:
            StrindForSave += str(round(ResPredictions[i], 4)) + ' '
        elif ResPredictions[i] < 150:
            StrindForSave += str(round(ResPredictions[i], 3)) + ' '
        else:
            StrindForSave += str(round(ResPredictions[i], 1)) + ' '
    with open(SaveFileName, 'r') as f:
        l = f.readlines()
    with open(SaveFileName, 'w') as f:
        StrindForSave = str(PredictionNumber) + ' ' + CryptoName[:CryptoName.index(".")] + ' : ' + StrindForSave + '\n'
        #print(StrindForSave)
        f.write(StrindForSave)
        f.writelines(l)
def MainPrediction(PredictionNumberOfDay = 0, PredictPeriod = 3, CryptoNames = BaseConfig.CryptoNames):
    #PredictionNumberOfDay  вход данные
    #PredictPeriod период до которого идут предсказания
    # CryptoNames = ['EGLD.csv']
    for NameDB in CryptoNames:

        if NameDB in BaseConfig.NoUse or NameDB not in CryptoNames:
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
        massive_input = np.array(joined.iloc[PredictionNumberOfDay][1:])
        #Получение прогнозов
        ResPredictions = AIModel(massive_input, joined, PredictPeriod)
        #Показ прогнозов
        #ShowResult(ResPredictions, NameDB)
        SaveResults(ResPredictions[0], NameDB, PredictionNumberOfDay)
    SaveTime()
def ProPrediction(PredictionNumberOfDay = 0, PredictPeriod = 3, CryptoNames = BaseConfig.CryptoNames, SaveFileName = 'SaveResults.txt', AIFunction = AIModelPro, Return = False):
    #PredictionNumberOfDay  вход данные
    #PredictPeriod период до которого идут предсказания
    for NameDB in CryptoNames:
        if NameDB in BaseConfig.NoUse: # or NameDB not in CryptoNames
            continue
#        if PeriodDayBefor >= 90:
#            CryptoNames = BaseConfig.ExtraClass
#            CryptoNames.extend(BaseConfig.SuperExtraClass)
        df = pd.read_csv('DataBase/' + NameDB)
        # Очишение баз данных
        CleaininDF(df)
        #Получение прогнозов
        ResPredictions = AIFunction(df, PredictPeriod, NameDB, PredictionNumberOfDay)
        #print(ResPredictions)
        #ShowResult(ResPredictions, NameDB)

        #print(ResPredictions[0:1] + ResPredictions[1::3])
        SaveResults(ResPredictions[0:1] + ResPredictions[1::3] + ResPredictions[2:4], NameDB, PredictionNumberOfDay, SaveFileName=SaveFileName)
    SaveTime(SaveFileName)
    if Return is True:
        return ResPredictions

'''
Сделать изменение колва прошлых дней для каждой монеты отдельно
Сделать показ предыдущего графика
Сделать функционализацию +
Сделать повтор анализа и показ на графике
Сделать запись файлов в текстовый документ
Показ среднего арифметического 
Сделать легендарный средний и плохой по прогнозам списки и из ввывод
'''
