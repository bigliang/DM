import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_csv("sales_train.csv")

scaler = MinMaxScaler(feature_range=(-1, 1))
data['date'] = pd.to_datetime(data['date'], format = '%d.%m.%Y')
ori_data=data
data = data.groupby('date').sum()['item_cnt_day']
ori_data=ori_data.groupby('date').sum()['item_cnt_day']

# test stationarity with adfuller test and draw rolling average picture
def draw_rolling_average(data):
    data_mean=pd.rolling_mean(data,window=30)
    data_std=pd.rolling_std(data,window=30)
    plt.plot(data,label='Original')
    plt.plot(data_mean, label='Rolling Mean')
    plt.plot(data_std, label = 'Rolling Std')
    plt.legend()
    plt.show()

def DF_test(data):
    result = adfuller(data,autolag='AIC')
    print('Test Statistic:',result[0])
    print('p-value:',result[1])
    print('Lags Used:',result[2])
    print('Observations Used:',result[3])
    for key,value in result[4].items():
        print('Critical Value (%s)'%key, value)

# find the best p q
def test_p_q(data):
    pmax = int(len(data_log)/100)
    qmax = int(len(data_log)/100)
    bic_matrix = []
    for p in range(pmax+1):
      tmp = []
      for q in range(qmax+1):
        try:
          tmp.append(ARIMA(data_log, (p,1,q)).fit().bic)
        except:
          tmp.append(None)
      bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)
    p,q = bic_matrix.stack().idxmin()
    print(u'BIC最小的p值和q值为：%s、%s' %(p,q))

# draw grap
def Draw_pre(newres):
    plt.figure(figsize=(40,10))
    plt.plot(ori_data,color='green',label='original')
    plt.plot(newres,color='black',label='predict')
    plt.legend()
    plt.show()

# draw test error
def draw_test(data,predict):
    plt.plot(data['2015-10-01':'2015-10-30'],label='true')
    ownnewres=predict
    ownnewres = pd.DataFrame(ownnewres)
    ownnewres.index=pd.date_range('2015-10-01','2015-10-30')
    plt.plot(ownnewres, label='predict')
    plt.title('MSE: %.4f'%(((data['2015-10-01':'2015-10-30'].values-ownnewres.values)**2).sum()/data_log.size))
    plt.legend()
    plt.show()
draw_rolling_average(data)
DF_test(data)
#plot ACF and PACF
plot_acf(data)
plt.show()

plot_pacf(data)
plt.show()
#log-transform
data_log=np.log(data)
data_log_scaled = scaler.fit_transform(data_log.reshape(1, -1))
plt.plot(data)
plt.show()

draw_rolling_average(data_log)
DF_test(data_log)
plot_acf(data_log)
plt.show()

#moving average
moving_average = pd.rolling_mean(data,30)
plt.plot(data)
plt.plot(moving_average, color='red')
plt.show()

avg_diff = data - moving_average
avg_diff.dropna(inplace=True)
draw_rolling_average(avg_diff)
DF_test(avg_diff)
plot_acf(avg_diff)
plt.show()

plot_pacf(avg_diff)
plt.show()

#EWMA
avg_weight = pd.ewma(data_log, halflife=12)
plt.plot(data_log)
plt.plot(avg_weight, color='red')
plt.show()

ewma_diff = data - avg_weight
draw_rolling_average(ewma_diff)
DF_test(ewma_diff)
plot_acf(ewma_diff)
plt.show()
plot_pacf(ewma_diff)
plt.show()

#decompose seasonarity
data_diff = data_log - data_log.shift()
plt.plot(data_diff)
plt.show()

data_diff.dropna(inplace=True)
draw_rolling_average(data_diff)
DF_test(data_diff)
plot_acf(data_diff)
plt.show()

plot_pacf(data_diff)
plt.show()


result = acorr_ljungbox(data_log, lags=1)

model = ARIMA(data_log, (9,1,8)).fit(disp=-1)
res=model.forecast(30)[0]
newres=np.exp(res)
newres = pd.DataFrame(newres)
newres.index=pd.date_range('2015-10-01','2015-10-30')
Draw_pre(newres)

plt.plot(data_log,label='true')
plt.plot(model.fittedvalues, label='predict')
plt.legend()
plt.title('MSE: %.4f'%(((np.exp(model.fittedvalues)-np.exp(data_log))**2).sum()/data_log.size))
plt.show()

predict = model.predict()
MAE = (np.abs(np.exp(predict)-np.exp(data_log))).sum()/data_log.size
print('data_log MAE',MAE)

draw_test(np.exp(data_log),np.exp(res))

model = ARIMA(avg_diff, (3,0,2)).fit()
plt.plot(avg_diff,label='true')
plt.plot(model.fittedvalues, label='predict')
plt.title('MSE: %.4f'%(((model.fittedvalues-avg_diff)**2).sum()/data_log.size))
plt.legend()
plt.show()

res=model.forecast(30)[0]
draw_test(avg_diff,res)

predict = model.predict()
# predict =np.exp(predict)
MAE = (np.abs(predict-avg_diff)).sum()/data.size
print('ma_MAE:',MAE)
# test_p_q(ewma_diff)
# EWMA
ewma_diff.dropna(inplace=True)
model = ARIMA(ewma_diff,(3,0,2)).fit()
plt.plot(ewma_diff,label='true')
plt.plot(model.fittedvalues, label='predict')
plt.title('MSE: %.4f'%(((model.fittedvalues-ewma_diff)**2).sum()/data_log.size))
plt.legend()
plt.show()

res=model.forecast(30)[0]
draw_test(ewma_diff,res)

res=model.forecast(30)[0]
newres=res
newres = pd.DataFrame(newres)
newres.index=pd.date_range('2015-10-01','2015-10-30')
Draw_pre(newres)

predict = model.predict()
MAE = (np.abs(predict-ewma_diff)).sum()/data.size
print('EWMA_MAE:',MAE)
