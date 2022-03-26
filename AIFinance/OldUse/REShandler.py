import pandas as pd
#from AIFinance import InputRefresh
mode = 'p' #input("Input mode:\n").lower()
monet = ['BTC', 'ETH', 'BNB', 'LINK', 'DASH', 'DOT', 'LTC', 'QTUM', 'XRP', 'SOL', 'AVA', 'VET', 'CAKE', 'ADA', 'SUSHI', 'OMG']
MonetBasePath = [
'DataBase/BitcoinInvesting.csv',
'DataBase/BNBInvesting.csv',
'DataBase/LINKInvesting.csv',
'DataBase/DASHInvesting.csv',
'DataBase/DOTInvesting.csv',
'DataBase/LTCInvesting.csv',
'DataBase/QTUMInvesting.csv',
'DataBase/XRPInvesting.csv',
'DataBase/SOLInvesting.csv',
'DataBase/AVAXInvesting.csv',
'DataBase/VETInvesting.csv',
'DataBase/CAKEInvesting.csv',
'DataBase/ADAInvesting.csv',
'DataBase/SUSHIInvesting.csv',
'DataBase/OMGInvesting.csv',
]
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

def GenerateProcent(NumbersOfLines):
    for MonetIndex in range(len(MonetBasePath)):
        df = pd.read_csv(MonetBasePath[MonetIndex])
        #print(df.head(10))
        with open('results.txt', "r") as file:
            text = file.readlines()
        for i in range(len(text)):
            if text[i][:-2] in monet:
                NeedStrings = text[i+1: i+NumbersOfLines]
                i += NumbersOfLines
                RetrStrings = [[], [], [], [], [], [], []]
                for string in NeedStrings:
                    time_list = InputRefresh(list(string.split('"')))
                    print(time_list)
                    for j in range(len(time_list)):
                        RetrStrings[j].append(time_list[j])
                #print(11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111)
                NewInfoDF = pd.DataFrame()#, index=['Дата', 'Цена', 'Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм. %'])
                #print(NewInfoDF)
                #print(11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111)
                break


if mode == 'p':
    GenerateProcent(6 + 1)
elif mode == 'q': # FOr auto analis
    pass
else:
    with open('results.txt', mode) as f:
        if mode == 'r':
            lines = f.readlines()
            print(lines)
        elif mode == "w":
            NumbersOfLines = int(input('Количество строк:\n'))
            for i in range(len(monet)):
                f.writelines(monet[i])
                for j in range(NumbersOfLines):
                    f.writelines(input())


