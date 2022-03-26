import math
import AIFinance
import BaseConfig
import pandas as pd
import numpy as np
import UpdateDataBase
import datetime as dt
import time
import binance
from Crypto import Crypting
from binance.enums import *

client = binance.Client(BaseConfig.ShifrKey, BaseConfig.ShifrSecretKey)
#account = client.get_account()

WasTrade = 0
Bag = {}
'''
Сделать альфа-проверку 
3 секторную
проверку множество раз 
с огранисением больше половины
и подсчёта количества 

Сделать многоуровневую
систему тестирования
+1 +0.5 +0.25
-1 -0.5 -0.25
'''
def GetMassiveInputs(OpenFileName = 'ResultsForTrades.txt'):
    with open(OpenFileName, 'r') as f:
        l = f.readlines()
    AllInputs = AIFinance.ClassSort(OpenFileName)
    res = AllInputs[0]
    res.extend(AllInputs[1])
    res.extend(AllInputs[2])
    for i in range(len(res)):
        res[i] = res[i].split()
    return res

def AlphaTradeCheck(Iteration = 5, BayFlag = 1.025, SellFlag = 1, PredictionNumberOfDay = 0, SaveFileName = 'ResultsForTrades.txt'):
    d = {}
    for i in range(Iteration):
        Massssive_WasInAnalis = []
        AIFinance.ProPrediction(PredictionNumberOfDay, CryptoNames=BaseConfig.TradeList, SaveFileName=SaveFileName)
        Massive_Inputs = GetMassiveInputs(SaveFileName)
        for massive in Massive_Inputs:
            if massive[1] in Massssive_WasInAnalis:
                continue
            Counter = 1
            SellPrices = list(map(float, [massive[4], massive[5]])) # , massive[6]
            for SellPrice in SellPrices:
                if SellPrice > float(massive[3]) * BayFlag:
                    if massive[1] in d.keys():
                        d[massive[1]] += Counter
                    else:
                        d[massive[1]] = Counter
                elif SellPrice < float(massive[3]) * SellFlag:
                    if massive[1] in d.keys():
                        d[massive[1]] -= Counter
                    else:
                        d[massive[1]] = -Counter
                else:
                    pass
                Counter /= 2

            Massssive_WasInAnalis.append(massive[1])
    return d

def AlphaСoefficient(dict):
    AlphaCoeff = 0
    for i in dict.keys():
        AlphaCoeff += dict[i]
    AlphaCoeff /= len(dict.keys())
    print('AlphaCoeff:', AlphaCoeff)
    if AlphaCoeff > 3:
        AlphaCoeff = 3
    elif AlphaCoeff < -3:
        AlphaCoeff = -3
    return AlphaCoeff

def SectorSolution(sectors = 3):
    time_res = []
    for i in range(sectors):
        time_res.append(AlphaTradeCheck(Iteration=7, BayFlag=1.012, SellFlag=1))
    Results = {}
    for i, j in time_res[0].items():
        sum = 0
        Error_Counter = 0
        for k in range(sectors):
            try:
                sum += time_res[k][i]
            except:
                Error_Counter += 1
        Results[i] = round(sum / (sectors - Error_Counter), 2)
    AlphaCoeff = AlphaСoefficient(Results)
    for i in Results.keys():
        Results[i] += AlphaCoeff
    print(Results)
    return Results

def DampOrPamp(DampFlag = -1.5, PampFlag = 1.5):
    procent = 0
    counter = 0
    for MonetName in BaseConfig.TradeList:
        DB = pd.read_csv('DataBase/' + MonetName)
        procent += float('.'.join(DB.loc[0]['Изм. %'][:-1].split(',')))
        #procent += float('.'.join(DB.loc[1]['Изм. %'][:-1].split(',')))
        counter += 1
    procent /= counter
    print('Procent: ', procent)
    if procent < DampFlag:
        return -1
    elif procent > PampFlag:
        return 1
    else:
        return 0

def PersonalAlphaCoefficent(MonetName):
    DB = pd.read_csv('DataBase/' + MonetName + '.csv')
    procent = float('.'.join(DB.loc[0]['Изм. %'][:-1].split(',')))
    return procent

#print(AlphaСoefficient({'MANA': 1.17, 'ZEC': -4.83, 'NEO': -9.83, 'XLM': -1.0, 'OMG': -10.0, 'ADA': -9.67, 'VET': -9.5, 'QTUM': -9.83, 'LTC': -8.67, 'DASH': 8.83, 'LINK': -8.0, 'BNB': -6.33, 'TRX': -7.5, 'BNT': -3.5, 'BCH': -6.33, 'RVN': -6.83, 'XRP': 3.5, 'BTC': -4.83, 'ATOM': -8.5, 'CHZ': -8.83, 'MATIC': -2.83, 'IOTA': -3.33, 'ETH': -3.0}))
#print(AlphaTradeCheck())
#print(SectorSolution())

def BagUpdate():
    for i in range(len(BaseConfig.TradeList)):
        MonetName = BaseConfig.TradeList[i][:BaseConfig.TradeList[i].index('.')]
        Bag[MonetName + '.csv'] = client.get_asset_balance(MonetName)
    print(Bag)

#{'MANA': 2.858695652173913, 'ZEC': -6.141304347826087, 'NEO': -11.641304347826086, 'XLM': -4.641304347826087, 'OMG': -13.141304347826086, 'ADA': -12.641304347826086, 'VET': -13.141304347826086, 'QTUM': -13.141304347826086, 'LTC': -13.141304347826086, 'DASH': -0.14130434782608692, 'BNB': -6.641304347826087, 'TRX': -7.641304347826087, 'BNT': -10.641304347826086, 'BCH': -7.141304347826087, 'RVN': -4.141304347826087, 'IOTA': -8.141304347826086, 'XRP': -4.141304347826087, 'BTC': -10.141304347826086, 'ATOM': -12.641304347826086, 'CHZ': -8.641304347826086, 'MATIC': -6.141304347826087, 'LINK': -7.641304347826087, 'ETH': -3.641304347826087}

def SecondBrain(PredictionNumberOfDay = 0, TradeList=BaseConfig.TradeList, FileNameForTrades = 'ResultsForTrades.txt', SellFlag = 1, BayFlag = 3):
    global WasTrade
    WasTrade += 1

    print('Update')
    UpdateDataBase.MainUpdate()
    print('Update finished')

    DictPredict = SectorSolution(sectors=1)
    PorD = DampOrPamp(DampFlag=-0.9, PampFlag=0.9)
    if PorD == -1:
        print('Damp')
        BayAndSellDict = BaseConfig.BayAndSellDictDamp
    elif PorD == 1:
        print('Pamp')
        BayAndSellDict = BaseConfig.BayAndSellDictPamp
    else:
        print('Normal State')
        time.sleep(30)
        return None
        #BayAndSellDict = BaseConfig.BayAndSellDictUsuall

    Massive_Inputs = GetMassiveInputs(FileNameForTrades)
    Massssive_WasInAnalis = []
    for massive in Massive_Inputs:
        try:
            PriceDollar = float(client.get_avg_price(symbol='USDTRUB')['price'])
            if massive[1] in Massssive_WasInAnalis:
                continue
            MonetProcent = PersonalAlphaCoefficent(massive[1])
            if DictPredict[massive[1]] >= BayAndSellDict[massive[1]][0] and 10 > MonetProcent > 1 or MonetProcent < -4:
                try:
                    print(massive[1], "ForBay")
                    if massive[1] + '.csv' in BaseConfig.NoTradeList:
                        print(massive[1], "NoForTrading")
                        continue
                    AVG_Info = client.get_avg_price(symbol=massive[1] + 'USDT')
                    MoneyForBay = float(client.get_asset_balance('USDT')['free']) / 2.3
                    #MoneyForBay = 13

                    MonetBalance = float(client.get_asset_balance(massive[1])['free'])

                    quantity_1 = MoneyForBay / float(AVG_Info['price'])
                    if massive[1] in BaseConfig.MonetBorder.keys():
                        if MonetBalance * float(AVG_Info['price']) + MoneyForBay > BaseConfig.MonetBorder[massive[1]] / PriceDollar:
                            quantity_1 = BaseConfig.MonetBorder[massive[1]] / PriceDollar / float(AVG_Info['price']) - MonetBalance
                            MoneyForBay = quantity_1 * float(AVG_Info['price'])
                            print('Was use border', 'MoneyForBay -', MoneyForBay, 'quantity -', quantity_1)

                    if float(AVG_Info['price']) > 50:
                        quantity_1 = round(quantity_1, 2)
                    elif float(AVG_Info['price']) > 5:
                        quantity_1 = round(quantity_1, 1)
                    else:
                        quantity_1 = int(quantity_1)
                    print(massive[1], quantity_1)
                    if MoneyForBay >= 10:
                        OrderBay = client.order_market_buy(
                            symbol=massive[1] + 'USDT',
                            quantity=quantity_1
                        )
                except Exception as E:
                    print(massive[1], "Error_0")
                    print('Ошибка:\n', E)
            elif DictPredict[massive[1]] <= BayAndSellDict[massive[1]][1]:
                try:
                    print(massive[1], 'ForSale')
                    AVG_Info = client.get_avg_price(symbol=massive[1] + 'USDT')
                    quantity_1 = float(client.get_asset_balance(massive[1])['free'])
                    if float(AVG_Info['price']) > 50:
                        quantity_1 = math.floor(quantity_1 * 100) / 100.0
                    elif float(AVG_Info['price']) > 3.5:
                        quantity_1 = math.floor(quantity_1 * 10) / 10.0
                    else:
                        quantity_1 = int(math.floor(quantity_1 * 1) / 1.0)
                    if quantity_1 > 0:
                        print(massive[1], quantity_1)
                        OrderSell = client.order_market_sell(
                            symbol=massive[1] + 'USDT',
                            quantity=quantity_1
                        )
                except Exception as E:
                    print(massive[1], "Error_1")
                    print('Ошибка:\n', E)
            else:
                print(massive[1], "WasAnalis")
            Massssive_WasInAnalis.append(massive[1])
        except Exception as E:
            print('Main Error', E)
            pass
    print(dt.datetime.now())

#AVG_Info = client.get_avg_price(symbol='ETHUSDT')
#print(AVG_Info)
#print(client.get_asset_balance("ETH"))
def Main_Start(Flag = False):
    BaseConfig.TradeMode = True
    while BaseConfig.TradeMode == True:
        #FirstBrain(0)
        if BaseConfig.TradeMode == True:
            print('Bot work continue ')
            SecondBrain(0)
        else:
            print('Bot was stoped !!!')
            break
        #print(Bag)
        #time.sleep(60 * 0.3)

def Stop():
    BaseConfig.TradeMode = False
