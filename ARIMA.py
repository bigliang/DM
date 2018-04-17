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
#scaler = StandardScaler()
scaler = MinMaxScaler(feature_range=(-1, 1))
data['date'] = pd.to_datetime(data['date'], format = '%d.%m.%Y')
ori_data=data
data = data.groupby('date').sum()['item_cnt_day']
ori_data=ori_data.groupby('date').sum()['item_cnt_day']

#TEST stationarity with adfuller test and draw rolling average picture
def draw_rolling_average(data):
    data_mean=pd.rolling_mean(data,window=30)
    data_std=pd.rolling_std(data,window=30)
    plt.plot(data,label='Original')
    plt.plot(data_mean, label='Rolling Mean')
    plt.plot(data_std, label = 'Rolling Std')
    plt.legend()
    plt.show()
def DF_test(data): #adfuller test
    result = adfuller(data,autolag='AIC')
    print('Test Statistic:',result[0])
    print('p-value:',result[1])
    print('Lags Used:',result[2])
    print('Observations Used:',result[3])
    for key,value in result[4].items():
        print('Critical Value (%s)'%key, value)
#find the best p q
def test_p_q(data):
    pmax = int(len(data_log)/100)# the max p , q will no more than 100
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
                p,q = bic_matrix.stack().idxmin()#find the minimum BIC index
    print(u'Smallest BIC P Q is：%s、%s' %(p,q))

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
    plt.legend()
    plt.show()
plt.title("original data rolling average")
draw_rolling_average(data)
DF_test(data)

#plot ACF and PACF
plot_acf(data)
plt.title("original data acf")
plt.show()

plot_pacf(data)
plt.title("original data pacf")
plt.show()
#log-transform
data_log=np.log(data)
plt.plot(data)
plt.title("log-transformed data")
plt.show()

plt.title("log-transformed data rolling average")
draw_rolling_average(data_log)
DF_test(data_log)

plot_acf(data_log)
plt.title("log-transformed data acf")
plt.show()

plot_pacf(data_log)
plt.title("log-transformed data pacf")
plt.show()

#moving average
moving_average = pd.rolling_mean(data,30)# moving average window sie = 30
plt.plot(data)
plt.plot(moving_average, color='red')
plt.title("moving average")
plt.show()

avg_diff = data - moving_average
avg_diff.dropna(inplace=True)
plt.title("moving average rolling average")
draw_rolling_average(avg_diff)
DF_test(avg_diff)


plot_acf(avg_diff)
plt.title("moving average acf graph")
plt.show()

plot_pacf(avg_diff)
plt.title("moving average pacf graph")
plt.show()

#EWMA
avg_weight = pd.ewma(data_log, halflife=12)
plt.plot(data_log)
plt.plot(avg_weight, color='red')
plt.title("weighted moving average ")
plt.show()

ewma_diff = data - avg_weight
plt.title("weighted moving average rolling average ")
draw_rolling_average(ewma_diff)
DF_test(ewma_diff)
plot_acf(ewma_diff)
plt.title("weighted moving average acf")
plt.show()
plot_pacf(ewma_diff)
plt.title("weighted moving average pacf")
plt.show()

#decompose seasonarity
data_diff = data_log - data_log.shift()
plt.title("difference data")
plt.plot(data_diff)
plt.show()

data_diff.dropna(inplace=True)
plt.title("dfieerence data")
draw_rolling_average(data_diff)
DF_test(data_diff)

plot_acf(data_diff)
plt.title("difference data acf")
plt.show()

plot_pacf(data_diff)
plt.title("difference data pacf")
plt.show()


result = acorr_ljungbox(data_log, lags=1)#white nosie test

model = ARIMA(data_log, (2,0,8)).fit(disp=-1)#train the model for log-transformed data
res=model.forecast(30)[0]
newres=np.exp(res)
newres = pd.DataFrame(newres)
newres.index=pd.date_range('2015-10-01','2015-10-30')
Draw_pre(newres)

# plt.plot(data_log,label='true')
# plt.plot(model.fittedvalues, label='predict')
# plt.title("training reuslt for Log-transformed data")
# plt.legend()
# plt.show()

predict = model.predict()
train_MSE = ((np.exp(predict)-np.exp(data_log))**2).sum()/data.size
train_RMSE = np.sqrt(((np.exp(predict)-np.exp(data_log))**2).sum()/data.size)
train_MAE= (np.abs(np.exp(predict)-np.exp(data_log)).sum())/data.size
train_MAPE=(np.abs(np.exp(predict)-np.exp(data_log))*100/np.exp(data_log)).sum()/data.size
test_MAE = (np.abs(np.exp(predict)-np.exp(data_log))).sum()/data_log.size
test_MSE = ((np.exp(res)-np.exp(data_log['2015-10-01':'2015-10-30']))**2).sum()/data['2015-10-01':'2015-10-30'].size
test_RMSE = np.sqrt(((np.exp(res)-np.exp(data_log['2015-10-01':'2015-10-30']))**2).sum()/data['2015-10-01':'2015-10-30'].size)
test_MAE= (np.abs(np.exp(res)-np.exp(data_log['2015-10-01':'2015-10-30'])).sum())/data['2015-10-01':'2015-10-30'].size
test_MAPE=(np.abs(np.exp(res)-np.exp(data_log['2015-10-01':'2015-10-30']))*100/np.exp(data_log['2015-10-01':'2015-10-30'])).sum()/data['2015-10-01':'2015-10-30'].size

draw_test(np.exp(data_log),np.exp(res))#draw test result for log-transformed data

# train model for the Moving average
model = ARIMA(avg_diff, (3,0,4)).fit()
plt.plot(avg_diff,label='true')
plt.plot(model.fittedvalues, label='predict')
plt.title("training reuslt for MA data")
plt.legend()
plt.show()

res=model.forecast(30)[0]
plt.title("test reuslt for MA data")
draw_test(avg_diff,res)

predict = model.predict()
#predict =np.exp(predict)
# compute the loss
MAE = (np.abs(predict-avg_diff)).sum()/data.size
train_MSE = ((predict-avg_diff)**2).sum()/data.size
train_RMSE = np.sqrt(((predict-avg_diff)**2).sum()/data.size)
train_MAE= (np.abs(predict-avg_diff)).sum()/data.size
train_MAPE=(np.abs(predict-avg_diff)*100/ewma_diff).sum()/data.size
test_MSE = ((res-avg_diff['2015-10-01':'2015-10-30'])**2).sum()/data['2015-10-01':'2015-10-30'].size
test_RMSE = np.sqrt(((res-avg_diff['2015-10-01':'2015-10-30'])**2).sum()/data['2015-10-01':'2015-10-30'].size)
test_MAE= (np.abs(res-avg_diff['2015-10-01':'2015-10-30'])).sum()/data['2015-10-01':'2015-10-30'].size
test_MAPE=(np.abs(res-avg_diff['2015-10-01':'2015-10-30'])*100/ewma_diff['2015-10-01':'2015-10-30']).sum()/data['2015-10-01':'2015-10-30'].size


#test_p_q(ewma_diff)
#EWMA
ewma_diff.dropna(inplace=True)
model = ARIMA(ewma_diff,(3,0,18)).fit()
plt.plot(ewma_diff,label='true')
plt.plot(model.fittedvalues, label='predict')
plt.title("training reuslt for WMA data")
plt.legend()
plt.show()

res=model.forecast(30)[0]
plt.title("test reuslt for WMA data")
draw_test(ewma_diff,res)

res=model.forecast(30)[0]
newres=res
newres = pd.DataFrame(newres)
newres.index=pd.date_range('2015-10-01','2015-10-30')
Draw_pre(newres)
#compute loss
predict = model.predict()
MSE = ((predict-ewma_diff)**2).sum()/data.size
train_RMSE = np.sqrt(((predict-ewma_diff)**2).sum()/data.size)
train_MAE= (np.abs(predict-ewma_diff)).sum()/data.size
train_MAPE=(np.abs(predict-ewma_diff)*100/ewma_diff).sum()/data.size

test_MSE = ((res-ewma_diff['2015-10-01':'2015-10-30'])**2).sum()/data.size
test_RMSE = np.sqrt(((res-ewma_diff['2015-10-01':'2015-10-30'])**2).sum()/data.size)
test_MAE= (np.abs(res-ewma_diff['2015-10-01':'2015-10-30'])).sum()/data.size
test_MAPE=(np.abs(res-ewma_diff['2015-10-01':'2015-10-30'])*100/ewma_diff['2015-10-01':'2015-10-30']).sum()/data.size

