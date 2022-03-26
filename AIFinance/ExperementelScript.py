import pandas as pd
from binance import Client
from pandas import DataFrame

import BaseConfig
import PIL as p
from binance.helpers import round_step_size
import math
import matplotlib
import AIFinance
from matplotlib import pyplot as plt
import numpy as np
'''amount = 0.000234234
tick_size = 0.00001
rounded_amount = round_step_size(amount, tick_size)
print(rounded_amount)'''

def PrintBalance():
    client = Client(BaseConfig.ShifrKey, BaseConfig.ShifrSecretKey)
    print()
    account = client.get_account()
    balance = account['balances']
    PriceDollar = float(client.get_avg_price(symbol='USDTRUB')['price'])
    FreeBalance = client.get_asset_balance('USDT')
    print('Free USDT: ', float(FreeBalance['free']) * PriceDollar)
    print(PriceDollar)
    AllBalance = float(FreeBalance['free']) * PriceDollar
    for i in balance:
        if float(i['free']) > 0:
            try:
                NowPrice = float(client.get_avg_price(symbol=i['asset'] + 'USDT')['price'])
                BalanceOfMonetUSDT = float(i['free']) * NowPrice * PriceDollar
                print(i['asset'], NowPrice, i['free'], BalanceOfMonetUSDT)
                AllBalance += BalanceOfMonetUSDT
            except Exception as E:
                pass
                #print(i['asset'], E)
    print('AllBalance: ', AllBalance)

#PrintBalance()

def ProPrediction(df, PredictionNumberOfDay = 0, PredictPeriod = 3, CryptoNames = BaseConfig.CryptoNames, SaveFileName = 'SaveResults.txt', AIFunction = AIFinance.AIModelPro, Return = False):
    for NameDB in CryptoNames:
        if NameDB in BaseConfig.NoUse: # or NameDB not in CryptoNames
            continue
        # Очишение баз данных
        df = pd.DataFrame(df,  columns=['Цена'])
        AIFinance.CleaininDF(df)
        #Получение прогнозов
        ResPredictions = AIFunction(df, PredictPeriod, NameDB, PredictionNumberOfDay)
    return ResPredictions

def MainPredict(NameDB = 'BTC.csv', Comparison=True):
    df = pd.read_csv('DataBase/' + NameDB)
    arr = list(df['Цена'])

    if Comparison is True:
        ForLook = arr[:DayAmount + 1]
        arr = arr[DayAmount:]
        arr1 = arr[DayAmount:]

    for i in range(DayAmount):
        NewInfo = ProPrediction(arr, 0, CryptoNames=[NameDB], Return=True)
        NewInfo1 = ProPrediction(arr, 0, CryptoNames=[NameDB], Return=True)
        NewInfo2 = ProPrediction(arr, 0, CryptoNames=[NameDB], Return=True)
        PredictPrice = (NewInfo[1] + NewInfo1[1] + NewInfo2[1]) / 3
        if PredictPrice > 70:
            arr.insert(0, round(PredictPrice, 1))
        elif PredictPrice > 10:
            arr.insert(0, round(PredictPrice, 2))
        else:
            arr.insert(0, round(PredictPrice, 5))

        if Comparison is True:
            arr1.insert(0, ForLook[i])

    #fig, ax = plt.subplots(figsize=(1, 1))
    df = pd.DataFrame(arr,  columns=['Цена'])
    #print(df)
    AIFinance.CleaininDF(df)
    y = np.arange(0, 3, 0.05)
    res = arr[:DayAmount + 1]
    if Comparison is True:
        print("Real:   ", *ForLook[::-1])
        #print("Predict 1:", arr1)#в разработке
    print("Predict:", *res[::-1])

DayAmount = 100
df = pd.read_csv("DataBase\Stocks\AAPL.csv")
AIFinance.CleaininDF(df)
print(AIFinance.AIModelPro(DataFrame=df, PredictionNumberOfDay=0, PredictPeriod=3, MonetName='Stocks\AAPL.csv'))
for i in range(1):
    MonetName = 'Stocks/AAPL' + '.csv'
    if MonetName not in BaseConfig.NoUse:
        print()
        print(MonetName)
        MainPredict(MonetName, False)
        print()
        MainPredict(MonetName, True)

'''
import time
from UpdateDataBase import MainUpdate
while True:
    print('Now Update')
    MainUpdate()
    print('Now Sleep')
    time.sleep(60 * 60)
'''

#ax.plot(range(len(res)), res[::-1], linestyle='-', marker='o', linewidth=2, color='green', label='Цена')
#plt.show()
#ax.set_yticks(np.arange(1, 3, 1))
#ax.set_xticks(np.arange(0, 60, 5))
#ax.legend(loc='lower left')
#ax.set_title(NameDB[:-4], fontsize=16)
#ax.plot(range(50), arr[-100: -50], linestyle='-', marker='o', linewidth=2, color='green', label='Цена')
#ax.set_title(NameDB[:-4], fontsize=16)
#plt.show()


'''orders = client.get_all_orders(symbol='BNBUSDT', limit=10)
for order in orders:
    print(order)'''
#tickers = client.get_ticker()
#tickers = client.get_orderbook_tickers()
#for ticker in tickers:
#    print(ticker)
#trades = client.get_recent_trades(symbol='BNBUSDT')
#print(trades)
#trades = client.get_historical_trades(symbol='BNBUSDT')
#trades = client.get_aggregate_trades(symbol='BNBUSDT')
#for trade in trades:
#    print(trade)
#depth = client.get_order_book(symbol='BNBUSDT')
#print(depth)
'''status = client.get_system_status()
time_res = client.get_server_time()
print(status)
print(time_res)
print(client.ping())
info = client.get_account_snapshot(type='SPOT')
print(info)'''

#print(client.get_all_orders(''))
'''counter = 0
l = []
l1 = []
for i in range(1000000, 9999999, 2):
    flag = 0
    if '1' in str(i):
        flag += 1
    if '2' in str(i):
        flag += 1
    if '3' in str(i):
        flag += 1
    if '4' in str(i):
        flag += 1
    if '5' in str(i):
        flag += 1
    if '6' in str(i):
        flag += 1
    if '7' in str(i):
        flag += 1
    if '8' in str(i):
        flag += 1
    if '9' in str(i):
        flag += 1
    if '0' in str(i):
        flag += 1

    if flag == 2 and i % 2 == 0 and '0' in str(i):
        counter += 1
        print(i)
print(counter)
for i in range(len(l)):
    if int(l1[i]) != l[i]:
        print('Error -- ', l[i], l1[i])'''
'''
a = 0
b = 0
while a < 500:
    while b < 1000:
        if a*a + a == b*b + b + 1:
            print(a, b)
        a = round(a + 0.001, 3)
        b = round(b + 0.001, 3)
'''
'''for a in range(0, 50, 1):
    for b in range(0, 100, 1):
        print(a, b)
        if a*a + a == b*b + b + 1:
            print(a, b)'''



'''l = input().split()
print(int(l[1]) + int(l[0]))
n, m , k = int(input()), int(input()), int(input())
l1 = [] # не получилось без теории
l2 = []
for i in range(n):
    a = input().split()
    l1.append(int(a[1]))
    l2.append(int(a[2]))
res = []
for i in range(n):
    start = l1[i]
    for j in range(len(l2[i])):
        to = l2[i][j]'''

'''import tensorflow as tf
import keras as keras
from keras.layers.core import activation
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

Inputs = np.array([0.2, 0.3, 0.4])
Outputs = np.array([1.4, 1.6, 1.6])

model = keras.Sequential()
model.add(layers.Dense(units=1, activation='linear'))
model.compile(loss='mse', optimizer='sgd')
fit_results = model.fit(x=Inputs, y=Outputs, epochs=10)
predictions = fit_results.predict([24.5])
print(predictions)'''