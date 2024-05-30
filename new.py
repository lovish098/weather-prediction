import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing 

weather_df = pd.read_csv('kanpur.csv', parse_dates=['date_time'], index_col='date_time')


# print(weather_df.head(5))
# print(weather_df.columns)
# print(weather_df.shape)
# print(weather_df.describe())
# print(weather_df.isnull().any())


weather_df_num=weather_df.loc[:,['maxtempC','mintempC','cloudcover','humidity','tempC', 'sunHour','HeatIndexC', 'precipMM', 'pressure','windspeedKmph']]


print(weather_df_num.head())
print(weather_df_num.shape)
print(weather_df_num.columns)
# ploting values of all year
weather_df_num.plot(subplots=True, figsize=(25,20))



# plot the vaues of 1 year
weather_df_num['2019':'2020'].resample('D').fillna(method='pad').plot(subplots=True, figsize=(25,20))


weather_df_num.hist(bins=10,figsize=(15,15))


weth=weather_df_num['2019':'2020']
print(weth.head())

weather_y=weather_df_num.pop("tempC")
weather_x=weather_df_num

train_X,test_X,train_y,test_y=train_test_split(weather_x,weather_y,test_size=0.2,random_state=4)

print(train_X.shape)
print(train_y.shape)
print(train_y.head())


plt.scatter(weth.mintempC, weth.tempC)
plt.xlabel("Minimum Temperature")
plt.ylabel("Temperature")
plt.show()

plt.scatter(weth.HeatIndexC, weth.tempC)
plt.xlabel("Heat Index")
plt.ylabel("Temperature")
plt.show()

plt.scatter(weth.pressure, weth.tempC)
plt.xlabel("Minimum Temperature")
plt.ylabel("Temperature")
plt.show()

