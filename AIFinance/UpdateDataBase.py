from bs4 import BeautifulSoup
import requests
import codecs
import BaseConfig
from threading import Thread
import asyncio

def InputAntiRefresh(massive):
    ResStr = ''
    for i in massive:
        ResStr += '"' + i + '"' + ','
    return ResStr[:-1]
def SaveResults(StrindForSave , CryptoName):
    file = codecs.open('DataBase/' + CryptoName, "r", "utf-8")
    data = file.readlines()
    file.close()
    #print(int(data[1][1:3]))
    #print(int(StrindForSave[1:3]))
    '''print(StrindForSave[1:3])
    print(data[1][1:3])
    print(data[1], end='')
    print(data[2],end='')
    print(StrindForSave)'''
    if data[1][1:10] != StrindForSave[1:10] and data[2][1:10] != StrindForSave[1:10]: #It work, but No Work that because of 31 + 1 != 01  and int(data[1][1:3]) + 1 == int(StrindForSave[1:3])
        data1 = [data[0], StrindForSave + '\n'] + data[1:]
    elif int(data[1][1:3]) == int(StrindForSave[1:3]): # Was After if int(data[1][1:3]) == int(StrindForSave[1:3])
        data1 = [data[0], StrindForSave + '\n'] + data[2:]
    else:
        data1 = data

    file = codecs.open('DataBase/' + CryptoName, 'w', "utf-8")
    file.writelines(data1)
    file.close()
def Parse(URl, Day = 0):
    HEADERS = {'User-Agent': 'Mozilla/4.0 (Windows NT 7.0; Win32; x32; rv:85.0) Gecko/21010010 Firefox/85.0'}
    response = requests.get(URl, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    items = soup.find_all('tr')

    l = [[], [], [], [], [], [], []]
    counterForFinished = 0
    for i in items:
        i = i.find_all('td')
        try:
            if 'class="first left bold noWrap"' in str(i[0]):
                l[0].append(i[0].get_text())
                l[1].append(i[1].get_text())
                l[2].append(i[2].get_text())
                l[3].append(i[3].get_text())
                l[4].append(i[4].get_text())
                l[5].append(i[5].get_text())
                l[6].append(i[6].get_text())
            counterForFinished += 1
        except:
            pass

    res = l[0][Day], l[1][Day], l[2][Day], l[3][Day], l[4][Day], l[5][Day], l[6][Day]
    return res
def MainUpdate(CryptoNames = BaseConfig.CryptoNames, RecoverPeriod = 0):
    #CryptoNames = ['RVN.csv']
    #CryptoUrls = {'RVN': 'https://ru.investing.com/crypto/ravencoin/rvn-usd-historical-data'}
    RecoverPeriod = 1  # In the futers
    CryptoUrls = BaseConfig.CryptoUrls
    for CryptoName in CryptoNames:
        if CryptoName in BaseConfig.NoUse:
            continue
        try:
            #print(CryptoName, 'In Update',)
            CryptoUrl = CryptoUrls[CryptoName[:CryptoName.index('.')]]
            StrindForSaveNowDay = InputAntiRefresh(Parse(CryptoUrl))
            StrindForSaveLastDay = InputAntiRefresh(Parse(CryptoUrl, 1))
            SaveResults(StrindForSaveLastDay, CryptoName)
            SaveResults(StrindForSaveNowDay, CryptoName)
        except Exception as E:
            print(CryptoName, 'Error Update -- ', E)

def AsincoMainUpdate(CryptoNames = BaseConfig.CryptoNames, RecoverPeriod = 0):
    CryptoUrls = BaseConfig.CryptoUrls
    for CryptoName in CryptoNames:
        if CryptoName in BaseConfig.NoUse:
            continue
        try:
            CryptoUrl = CryptoUrls[CryptoName[:CryptoName.index('.')]]
            StrindForSaveNowDay = InputAntiRefresh(Parse(CryptoUrl))
            StrindForSaveLastDay = InputAntiRefresh(Parse(CryptoUrl, 1))
            SaveResults(StrindForSaveLastDay, CryptoName)
            SaveResults(StrindForSaveNowDay, CryptoName)
        except Exception as E:
            print(CryptoName, 'Error Update -- ', E)
#"22.12.2021","1,3345","1,2806","1,3667","1,2762","239,59M","4,22%"
#AsincoMainUpdate()