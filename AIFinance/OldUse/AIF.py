import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("BitCoin.csv")
'''print(df.head(10))
print(df.columns)
n = df.columns
print(df.shape)'''
#print(df.info())
for i in df.columns:
    if df[str(i)].dtype == "float64":
        df[str(i)] = df[str(i)].fillna(float(df[str(i)].mean()))
    elif df[str(i)].dtype == "int64":
        df[str(i)] = df[str(i)].fillna(df[str(i)].mean())
    else:
        df[str(i)] = df[str(i)].fillna(df[str(i)].describe(include=['object']).top)
lr = LinearRegression()
#t = df['btc_market_price'][1:]
x_train, x_test, y_train, y_test = train_test_split(df[0:-10].drop("Date", axis=1), df['btc_market_price'][10:], test_size = 0.05)
lr.fit(x_train, y_train)
predictions_y = lr.predict(x_train)
train_deviation = np.sqrt(mean_squared_error(y_train, predictions_y))
predictions_y_t = lr.predict(x_test)
test_deviation = np.sqrt(mean_squared_error(y_test, predictions_y_t))

k = [13852.92,16803200,2.33E+11,804914641.6,151529.1982,1.050657768,0,1516.419355,7.733333333,17165770.55,2.23E+12,32575141.38,491.3244648,1.256204351,143.1483801,592041,235045,292481273,224444,154354,1269172.799,193346.2866,2678410640]
isp = np.array(k)
sum = 0
count = 0
for i in range(100):
    sum += lr.predict(isp.reshape(1, -1))
    count += 1
print(13852.92)
print(*sum / count)
print(11282)
print(train_deviation, test_deviation)
#score = lr.score(x_test, y_test)
#print("score !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#print(score)