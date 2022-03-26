import pandas as pd

CleanlyCryptoNames = ['LUNA', 'IOTA', 'XLM', 'MATIC', 'RVN', 'BCH', 'SXP', 'CHZ', 'BNT', 'ONT', 'TRX', 'ETH', 'BTC', 'BNB', 'LINK', 'DASH', 'AVAX', 'DOT', 'LTC', 'QTUM', 'XRP', 'SOL', 'VET', 'CAKE', 'ADA', 'SUSHI', 'OMG', 'UNI', 'ETC', 'Stellar', 'XMR', 'EOS', 'CRO', 'EGLD', 'ATOM', 'MINA', 'SAND', 'NEO', 'ZEC', 'ETC', 'MANA', 'RUNE']

ClassNames = ['SuperExtraClass', 'ExtraClass', 'MidlClass', 'LowClass', 'ErrorClass']

SuperExtraClass = ['RVN.csv', 'BCH.csv', 'BNT.csv', 'TRX.csv', 'ETH.csv', 'BNB.csv', 'LINK.csv', 'DASH.csv', 'LTC.csv', 'QTUM.csv', 'VET.csv', 'ADA.csv', 'OMG.csv', 'XLM.csv', 'NEO.csv', 'ZEC.csv', 'MANA.csv']
ExtraClass =      ['BTC.csv', 'XRP.csv', 'IOTA.csv']
MidlClass =       ['SXP.csv', 'CHZ.csv', 'ONT.csv', 'ETC.csv', 'EOS.csv', 'CRO.csv', 'ATOM.csv', 'ETC.csv', 'MATIC.csv']
LowClass =        ['DOT.csv', 'SOL.csv', 'LUNA.csv']
ErrorClass =      ['AVAX.csv', 'CAKE.csv', 'SUSHI.csv', 'UNI.csv', 'XMR.csv', 'EGLD.csv', 'MINA.csv', 'SAND.csv', 'RUNE.csv']

TradeList = SuperExtraClass + ExtraClass + ['MATIC.csv', 'LUNA.csv', 'SOL.csv', 'CHZ.csv', 'ATOM.csv']
#TradeList = ['TRX.csv', 'MATIC.csv']
CryptoNames = (SuperExtraClass + ExtraClass + MidlClass + LowClass + ErrorClass).copy()

BadMSEMonet = ['MANA.csv', 'BCH.csv']# 'DASH.csv', 'NEO.csv','IOTA.csv', 'RVN.csv', 'OMG.csv', 'VET.csv'
TrdeBadMonet = [] #, 'ATOM.csv', 'ETH.csv', 'BNB.csv', 'ZEC.csv', 'MANA.csv'
NoInTestMonet = [] #'QTUM.csv', 'SOL.csv', 'CHZ.csv', 'LTC.csv', 'LUNA.csv'
NoLikeTradeMonet = ['RVN.csv']#'BNT.csv', 'XRP.csv'
NoTradeList = BadMSEMonet + TrdeBadMonet + NoInTestMonet + NoLikeTradeMonet

#MonetBorder = {'XRP': 3500, 'ATOM': 2500, 'SOL': 3500, 'QTUM': 6500, 'LTC': 4000, 'LUNA': 6500, 'MATIC': 4000, 'DASH': 4000, 'BNT': 2500, 'CHZ': 4000, 'ETH': 4000, 'BNB': 4000, 'ZEC': 4000} # В рублях, не забывай в доллар переводить
#MonetBorder = {'XRP': 5000, 'ATOM': 5500, 'SOL': 4500, 'QTUM': 5500, 'LTC': 4000, 'NEO': 3000, 'IOTA': 3000, 'LUNA': 7000, 'MATIC': 23000,
#               'DASH': 4000, 'BNT': 3000, 'CHZ': 8000, 'ETH': 3500, 'BNB': 5000, 'ZEC': 4000, 'TRX': 17000, 'RVN': 4000, 'OMG': 4000,
#               'VET': 4000, 'MANA': 2500, 'BCH': 2500, 'XLM': 4000} # В рублях, не забывай в доллар переводить
MonetBorder = {'XRP': 3500, 'ATOM': 2500, 'SOL': 2500, 'QTUM': 3500, 'LTC': 2500, 'NEO': 2000, 'IOTA': 3500, 'LUNA': 3500, 'MATIC': 4000,
               'DASH': 3500, 'BNT': 2500, 'CHZ': 4500, 'ETH': 1500, 'BNB': 2500, 'ZEC': 2500, 'TRX': 15000, 'RVN': 3500, 'OMG': 3000,
               'VET': 2500, 'MANA': 1000, 'BCH': 1500, 'XLM': 1500} # В рублях, не забывай в доллар переводить

TradeMode = False

print('Monet for trade:')
for i in TradeList:
    if i not in NoTradeList:
        print(i[:i.index('.')], end=' ')
print()

BayAndSellDictDamp = {
            'RVN':  [4.5, 1],
            'BCH':  [4.5, 1],
            # 'SXP' [1, -1
            'CHZ':  [7, 5.5], # на 6.5 и 5.5 продаёт много в -
            'BNT':  [7, 5.5],
            'ONT':  [3, 1],
            'TRX':  [6.5, 4.5],#'TRX':  [7, 5],
            'ETH':  [7, 5.5],
            'BTC':  [4, 2],
            'BNB':  [7, 5],
            'LINK': [7, 5],#'LINK': [-2, -3],
            'DASH': [5.5, 3.5],
            'AVAX': [4, 2],
            # 'DOT' [1, -1]
             'LTC':  [4, 2], # -6 , -8
            'QTUM': [5.5, 3.5],
            'XRP':  [5.5, 3.5],
            'SOL':  [5.5, 3.5],
            'VET':  [6,  5],
            # 'CAKE [1, -1
             'ADA':  [4, 2],
            # 'SUSH [1, -1
             'OMG':  [10, 8.5],
            # 'UNI' [1, -1
             # 'ETC' [1, -1]
             'XLM':  [6, 3.5],
            'XMR':  [4, 2],
            'EOS':  [4, 2],
            'CRO':  [4, 2],
            'EGLD': [4, 2],
            'ATOM': [7, 5],
            'MINA': [4, 2],
            'SAND': [4, 2],
            'NEO':  [6,  4],#'NEO':  [5,  3],
            'ZEC':  [11.5, 9],
            'ETC':  [5.5, 4],
            'MANA': [10, 9],
            'RUNE': [5.5, 4],
            'LUNA': [5.5, 4],
            'MATIC':[9, 8.5],#'MATIC': [2, 0],
            'IOTA': [8, 7]#'IOTA': [9, 5],
}
BayAndSellDictUsuall = {
            'RVN':  [5.5, 2],
            'BCH':  [5.5, 2],
            # 'SXP' [1, -1
            'CHZ':  [8, 5.5], # на 6.5 и 5.5 продаёт много в -
            'BNT':  [8, 5.5],
            'ONT':  [4, 1],
            'TRX':  [7.5, 4.5],#'TRX':  [7, 5],
            'ETH':  [8, 5.5],
            'BTC':  [5, 2],
            'BNB':  [8, 5],
            'LINK': [8, 5],#'LINK': [-2, -3],
            'DASH': [5.5, 3.5],
            'AVAX': [5, 2],
            # 'DOT' [1, -1]
             'LTC':  [5, 2], # -6 , -8
            'QTUM': [6.5, 3.5],
            'XRP':  [6.5, 3.5],
            'SOL':  [6.5, 3.5],
            'VET':  [7,  5],
            # 'CAKE [1, -1
             'ADA':  [5, 2],
            # 'SUSH [1, -1
             'OMG':  [11, 9],
            # 'UNI' [1, -1
             # 'ETC' [1, -1]
             'XLM':  [7, 3.5],
            'XMR':  [5, 2],
            'EOS':  [5, 2],
            'CRO':  [5, 2],
            'EGLD': [5, 2],
            'ATOM': [8, 4.5],
            'MINA': [5, 2],
            'SAND': [5, 2],
            'NEO':  [8,  6],#'NEO':  [5,  3],
            'ZEC':  [13, 9],
            'ETC':  [7, 3.5],
            'MANA': [11.5, 9],
            'RUNE': [7, 3.5],
            'LUNA': [7, 3.5],
            'MATIC':[10.5, 8.5],#'MATIC': [2, 0],
            'IOTA': [9.5, 7]#'IOTA': [9, 5],
}
BayAndSellDictPamp = {
            'RVN':  [3, 0],
            'BCH':  [3, 0],
            'CHZ':  [4, 2],
            'BNT':  [4, 2],
            'ONT':  [2, -1],
            'TRX':  [5.5, 1.5],
            'ETH':  [4, 0],
            'BTC':  [2, 0],
            'BNB':  [5, 3],
            'LINK': [4, 2],
            'DASH': [3, 1.5],
            'AVAX': [3, 0],
             'LTC':  [5, 2],
            'QTUM': [3.5, 0],
            'XRP':  [3, 0.5],
            'SOL':  [3.5, 0.5],
            'VET':  [4,  1],
             'ADA':  [3, 0],
             'OMG':  [7, 5],
             'XLM':  [6, 2.5],
            'XMR':  [4, 1],
            'EOS':  [3, 0],
            'CRO':  [3, 0],
            'EGLD': [3, 0],
            'ATOM': [5, 2.5],
            'MINA': [3, 0],
            'SAND': [3, 0],
            'NEO':  [6,  4],
            'ZEC':  [8, 6.5],
            'ETC':  [4.5, 1.5],
            'MANA': [8.5, 7],
            'RUNE': [5, 1.5],
            'LUNA': [5, 1.5],
            'MATIC':[7.5, 6],
            'IOTA': [6.5, 4.5]
}

def GenerateClearlyCryptoNames():
    CryptoNames1 = []
    for i in CryptoNames:
        CryptoNames1.append(i[:i.index('.')])
    return CryptoNames1
    print(CryptoNames1)

def LenForMonetBase():
    list = []
    ErrorClass = []
    ExtraClass = []
    SuperExtraClass = []
    MidlClass = []
    LowClass = []
    for NameDB in CryptoNames:
        df = pd.read_csv('DataBase/' + NameDB)
        list.append(NameDB)
        list.append(df.shape[0])
        if df.shape[0] < 450:
            ErrorClass.append(NameDB)
            #ErrorClass.append(df.shape[0])
        elif df.shape[0] < 600:
            LowClass.append(NameDB)
            #ErrorClass.append(df.shape[0])
        elif df.shape[0] > 2000:
            SuperExtraClass.append(NameDB)
            #SuperExtraClass.append(df.shape[0])
        elif df.shape[0] > 1000:
            ExtraClass.append(NameDB)
            #ExtraClass.append(df.shape[0])
        elif df.shape[0] > 600:
            MidlClass.append(NameDB)
            #MidlClass.append(df.shape[0])
    print(list)
    print(ErrorClass)
    print(SuperExtraClass)
    print(ExtraClass)
    print(MidlClass)
    print(LowClass)
    print(len(list))
    print(len(ErrorClass))
    print(len(SuperExtraClass))
    print(len(ExtraClass))
    print(len(MidlClass))
    print(len(LowClass))

def FunctionForGenerateFromCopyForCSV():
    a = ''
    l = []
    while a != "0":
        a = input()
        l.append(a)
    for i in l:
        print('"' + '","'.join(i.split()) + '"')

MonetErrorClassificator = {
              'RVN': 35,
              'BCH': 10,#
              # 'SXP'30,
              'CHZ': 25,
              'BNT': 105,
              'ONT': 35,
              'TRX': 15,
              'ETH': 5,
              'BTC': 5,
              'BNB': 5,
              'LINK':5,
              'DASH':15,
              'AVAX':15,
              # 'DOT'30,
              'LTC': 5, #
              'QTUM':150,
              'XRP': 25,
              'SOL': 5,
              'VET': 10,
              # 'CAKE30,
              'ADA': 5,
              # 'SUSH30,
              'OMG': 35,
              # 'UNI'30,
              # 'ETC'30,
              'XLM': 30,
              'XMR': 15, # По базам данных не проходит
              'EOS': 45,
              'CRO': 30,
              'EGLD':15,
              'ATOM':4,
              'MINA':5,
              'SAND':30,# Не проходит по Базе денных
              'NEO': 85,
              'ZEC': 25, # альтернатива 15
              'ETC': 15,
              'MANA': 15,
              'RUNE': 25,
              'LUNA': 5,
              'MATIC': 30,
              'IOTA': 15
              }

# MANA Pro 15 ideal
CryptoUrls = {'RVN': 'https://ru.investing.com/crypto/ravencoin/rvn-usd-historical-data',
              'BCH': 'https://ru.investing.com/crypto/bitcoin-cash/bch-usd-historical-data',
              # 'SXP': '',
              'CHZ': 'https://ru.investing.com/crypto/chiliz/chz-usd-historical-data',
              'BNT': 'https://ru.investing.com/crypto/bancor/bnt-usd-historical-data',
              'ONT': 'https://ru.investing.com/crypto/ontology/ont-usd-historical-data',
              'TRX': 'https://ru.investing.com/crypto/tron/trx-usd-historical-data',
              'ETH': 'https://ru.investing.com/indices/investing.com-eth-usd-historical-data',
              'BTC': 'https://ru.investing.com/crypto/bitcoin/btc-usd-historical-data',
              'BNB': 'https://ru.investing.com/crypto/binance-coin/bnb-usd-historical-data',
              'LINK': 'https://ru.investing.com/crypto/chainlink/link-usd-historical-data',
              'DASH': 'https://ru.investing.com/crypto/dash/dash-usd-historical-data',
              'AVAX': 'https://ru.investing.com/crypto/avalanche/avax-usd-historical-data',
              # 'DOT': '',
              'LTC': 'https://ru.investing.com/indices/investing.com-ltc-usd-historical-data',
              'QTUM': 'https://ru.investing.com/crypto/qtum/qtum-usd-historical-data',
              'XRP': 'https://ru.investing.com/indices/investing.com-xrp-usd-historical-data',
              'SOL': 'https://ru.investing.com/indices/investing.com-sol-usd-historical-data',
              'VET': 'https://ru.investing.com/crypto/vechain/ven-usd-historical-data',
              # 'CAKE': '',
              'ADA': 'https://ru.investing.com/indices/investing.com-ada-usd-historical-data',
              # 'SUSHI': '',
              'OMG': 'https://ru.investing.com/crypto/omg/omg-usd-historical-data',
              # 'UNI': '',
              # 'ETC': '',
              'XLM': 'https://ru.investing.com/crypto/stellar/historical-data',
              'XMR': 'https://ru.investing.com/crypto/monero/xmr-usd-historical-data',
              'EOS': 'https://ru.investing.com/crypto/eos/eos-usd-historical-data',
              'CRO': 'https://ru.investing.com/crypto/crypto-com-coin/cro-usd-historical-data',
              'EGLD': 'https://ru.investing.com/crypto/elrond-egld/erd-usd-historical-data',
              'ATOM': 'https://ru.investing.com/crypto/cosmos/atom-usd-historical-data',
              'MINA': 'https://ru.investing.com/crypto/mina/mina-usd-historical-data',
              'SAND': 'https://ru.investing.com/crypto/the-sandbox/sand-usd-historical-data',
              'NEO': 'https://ru.investing.com/crypto/neo/neo-usd-historical-data',
              'ZEC': 'https://ru.investing.com/crypto/zcash/zec-usd-historical-data',
              'ETC': 'https://ru.investing.com/crypto/ethereum-classic/etc-usd-historical-data?cid=1129153',
              'MANA': 'https://ru.investing.com/indices/investing.com-mana-usd-historical-data',
              'RUNE': 'https://ru.investing.com/crypto/thorchain/rune-usd-historical-data?cid=1165364',
              'LUNA': 'https://ru.investing.com/crypto/terra-luna/lunat-usd-historical-data',
              'MATIC': 'https://ru.investing.com/crypto/polygon/matic-usd-historical-data',
              'IOTA': 'https://ru.investing.com/crypto/iota/iota-usd-historical-data'
              }

NoUse = ['SXP.csv',
          'DOT.csv',
          'CAKE.csv',
          'SUSHI.csv',
          'UNI.csv',
          'SAND.csv']

MessageForHelp = '''
Это было в старой версии сейчас намного больше возможностей!!!
Умеет анализированть каждую монету по отдельность 
(Просто напиши / + "название монеты")
Умеет обновлять базу для каждой монеты или группы монет выборочно
(Просто напиши"название монеты(или группы монет)")
Умеет делать прогноз по всем монетам сразу
Для этого напиши /Прогноз
Умеет обновлять все базы данных сразу
Для этого напиши /Обновление_базы
Умеет выводить топ монет опираясь на предыдуший результат
Для этого напиши /top + кол-во монет(необезательный аргумент)
Но помни за свои средства отвечаешь только ты!!!
'''

ShifrKey = 'r13Je7OJrRYngLbDt7BtsXJuIgwbqb0nh2zP3jBdfcjvR4UaFHOha6Z1CoMkaprD'
ShifrSecretKey = 'i7OOV7jhirRKnBFaJYXYNv0BU7K6usYWAxtyW0ijMJF6BaMDplCCen7qdQ6gxYIX'
TelegramToken = '1793275901:AAF0G2aQyXeGcmbQbfGVQKTpgen1vt0_13Q'
