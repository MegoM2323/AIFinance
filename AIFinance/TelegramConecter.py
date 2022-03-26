import AIFinance
import UpdateDataBase
import BaseConfig
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import datetime as dt
import BinanceTradeBot
from threading import Thread

from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

bot = Bot(token=BaseConfig.TelegramToken)
dp = Dispatcher(bot)


UpdateLastTime = dt.datetime(2020, 11, 27,  19, 39)
delta = dt.timedelta(minutes=5)

btnPredict = KeyboardButton("/Прогноз")
btnTop = KeyboardButton("/top")
btnUpdate = KeyboardButton("/Обновление_базы")
btnPro = KeyboardButton("/Pro")
#btnPro_30 = KeyboardButton("/Pro")
#btnPro_60 = KeyboardButton("/Pro")

btnClass_Sort = KeyboardButton("/Class_Sort")
MainMarkUp = ReplyKeyboardMarkup().row(btnTop, btnUpdate, btnPredict).add(btnClass_Sort, btnPro)

CounterPrediction = 0
CryptoNames = BaseConfig.CleanlyCryptoNames

@dp.message_handler(commands=['start'])
async def process_start_command(message: types.Message):
    await message.reply("Привет, если что напиши - /help", reply_markup= MainMarkUp)

@dp.message_handler(commands=['help'])
async def help_process(message: types.Message):
    await message.reply(BaseConfig.MessageForHelp)

@dp.message_handler(commands=['C'])
async def process_counter_command(message: types.Message):
    await message.reply(CounterPrediction)

def Predict(PredictionNumberOfDay, PredictPeriod):
    AIFinance.MainPrediction(PredictionNumberOfDay, PredictPeriod)
    global CounterPrediction
    CounterPrediction += 1
    with open('SaveResults.txt', 'r') as f:
        res = f.readlines()
    ResStr = ''
    LenCryptoMonets = len(BaseConfig.CryptoNames) - len(BaseConfig.NoUse) + 1
    for i in res[:LenCryptoMonets]:
        ResStr += i
    return ResStr

@dp.message_handler(commands=['Прогноз'])
async def process_predict_0_command(message: types.Message):
    if len(message.text) > 8:
        PredictionNumberOfDay = message.text.split()[1]
    else:
        PredictionNumberOfDay = 0
    ResStr = Predict(int(PredictionNumberOfDay), 3)
    await message.reply(ResStr)

@dp.message_handler(commands=['Pro'])
async def process_Pro_command(message: types.Message):
    list = message.text.split()
    if len(list) > 1:
        MonetForScan = [list[1] + '.csv']
        if len(list) > 2:
            PredictionNumberOfDay_1 = int(list[2])
        else:
            PredictionNumberOfDay_1 = 0
    else:
        MonetForScan = BaseConfig.CryptoNames
        PredictionNumberOfDay_1 = 0
    AIFinance.ProPrediction(PredictionNumberOfDay = PredictionNumberOfDay_1, CryptoNames=MonetForScan, AIFunction=AIFinance.AIModelPro)
    global CounterPrediction
    CounterPrediction += 1
    with open('SaveResults.txt', 'r') as f:
        res = f.readlines()
    ResStr = ''
    if len(list) > 1:
        ResStr = res[1]
    else:
        LenCryptoMonets = len(BaseConfig.CryptoNames) - len(BaseConfig.NoUse) + 1
        #if PeriodDayBefor >= 90:
        #    LenCryptoMonets = len(BaseConfig.SuperExtraClass) + len(BaseConfig.ExtraClass)
        for i in res[:LenCryptoMonets]:
            ResStr += 'P' + i
    await message.reply(ResStr)

@dp.message_handler(commands=['Class_Sort'])
async def process_Class_Sort_command(message: types.Message):
    STR = ''
    res = AIFinance.ClassSort()

    for i in range(len(res)):
        STR += BaseConfig.ClassNames[i] + '\n'
        for j in range(len(res[i])):
            STR += res[i][j]
        STR += '\n'
    await message.reply(STR)

@dp.message_handler(commands=CryptoNames)
async def process_predict_OneMonet_command(message: types.Message):
    AIFinance.MainPrediction(0, 3, [message.text + '.csv'])
    global CounterPrediction
    CounterPrediction += 1
    with open('SaveResults.txt', 'r') as f:
        res = f.readlines()
    ResStr = res[1]
    await message.reply(ResStr)

@dp.message_handler(commands=['top'])
async def process_top_command(message: types.Message):
    res = AIFinance.SortLastInfo()
    if len(message.text) > 4:
        NumberOfMonet = int(message.text.split()[1])
    else:
        NumberOfMonet = len(res)
    ResStr = ''
    for i in res[:NumberOfMonet]:
        ResStr += i
    await message.reply(ResStr)

@dp.message_handler(commands=['Обновление_базы'])
async def process_updateDB_command(message: types.Message):
    global UpdateLastTime
    if dt.datetime.now() - UpdateLastTime > delta:
        await message.reply("Обновление запущено\n" + str(dt.datetime.now())[:-7])
        UpdateDataBase.MainUpdate()
        await message.reply("Обновлено\n" + str(dt.datetime.now())[:-7])
        UpdateLastTime = dt.datetime.now()
    else:
        await message.reply('База недавно была обновлена')

#@dp.message_handler(commands=['Bal'])
#async def process_get_balance_command(message: types.Message):
#    await message.reply(WasTradeInfo())


@dp.message_handler(content_types=types.ContentTypes.TEXT)
async def UpdateDataBase_Selectively(message: types.Message):
    try:
        user_id = message.from_user.id
        if user_id == 884712444:
            th = Thread(target=BinanceTradeBot.Main_Start)
            if message.text == 'c':
                await message.reply('Bot started')
                th.start()
            elif message.text == 's':
                BinanceTradeBot.Stop()
                try:
                    th.do_run = False
                except:
                    await message.reply('Bot Error Stopped')
                    print('Bot Stop Error')
                await message.reply('Bot stopped')
                print('Bot stopped')
            return
        else:
            await message.reply("Обновление запущено\n" + str(dt.datetime.now())[:-7])
            l = list(message.text.split())
            for i in range(len(l)):
                l[i] += '.csv'
            UpdateDataBase.MainUpdate(l)
            await message.reply("Обновлено\n" + str(dt.datetime.now())[:-7])
    except: pass


if __name__ == '__main__':
    executor.start_polling(dp)

