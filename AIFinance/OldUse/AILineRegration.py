import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("train.csv")
#print(df.head(10))
#print(df.columns)
n = df.columns
#print(df.shape)
#print(df.LotFrontage)
#print(df.info())
#df.LotFrontage = df.LotFrontage.fillna(df.LotFrontage.mean())
#df.MasVnrType = df.MasVnrType.fillna(df.MasVnrType.mean())
num_c = []
for i in n:
    if df[str(i)].dtype == "float64":
        df[str(i)] = df[str(i)].fillna(float(df[str(i)].mean()))
        num_c.append(str(i))
    elif df[str(i)].dtype == "int64":
        df[str(i)] = df[str(i)].fillna(df[str(i)].mean())
        num_c.append(str(i))
    else:
        df[str(i)] = df[str(i)].fillna(df[str(i)].describe(include=['object']).top)
print(num_c[:-1])
lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(df[num_c].drop('SalePrice', axis=1), df['SalePrice'], test_size=0.3 )
lr.fit(x_train[num_c[:-1]], y_train)
predictions_y = lr.predict(x_train[num_c[:-1]])
print(np.sqrt(mean_squared_error(y_train, predictions_y)))
predictions_y_t = lr.predict(x_test[num_c[:-1]])
print(np.sqrt(mean_squared_error(y_test, predictions_y_t)))
#score = lr.score(x_test, y_test)
#print("score !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#print(score)