import math
import AIFinance
from AIFinance import BaseConfig
import pandas as pd
import numpy as np
from AIFinance import UpdateDataBase
import datetime as dt
import time
import binance
from Crypto import Crypting
from binance.enums import *

client = binance.Client(BaseConfig.ShifrKey, BaseConfig.ShifrSecretKey)
#account = client.get_account()

WasTrade = 0
Bag = {}

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

def FirstBrain(PredictionNumberOfDay = 0, TradeList=BaseConfig.TradeList, FileNameForTrades = 'ResultsForTrades.txt'):
    global WasTrade
    WasTrade += 1
    print('Update')
    UpdateDataBase.MainUpdate()
    print('Update finished')
    AIFinance.ProPrediction(PredictionNumberOfDay, CryptoNames=TradeList, SaveFileName=FileNameForTrades)
    # Do predict Yesterday to improve
    Massive_Inputs = GetMassiveInputs(FileNameForTrades)
    Massssive_WasInAnalis = []
    for massive in Massive_Inputs:
        PriceDollar = float(client.get_avg_price(symbol='USDTRUB')['price'])
        if massive[1] in Massssive_WasInAnalis:
            continue
        massive[3] = float(massive[3])
        SellPrice = float(massive[4])
        if SellPrice > float(massive[3]) * 1.035:
            try:
                print(massive[1], "ForBay")
                if massive[1] + '.csv' in BaseConfig.NoTradeList:
                    print(massive[1], "NoForTrading")
                    continue
                AVG_Info = client.get_avg_price(symbol=massive[1] + 'USDT')
                MoneyForBay = float(client.get_asset_balance('USDT')['free']) / 2

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
        elif SellPrice < float(massive[3]) * 1:
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
    print(dt.datetime.now())