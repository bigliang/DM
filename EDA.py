import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns





items=pd.read_csv('items.csv')
sales_train=pd.read_csv('sales_train.csv')
data = pd.read_csv('sales_train.csv')
shops=pd.read_csv('shops.csv')

print(data.info())

print(data.dtypes)

sns.countplot(x='date_block_num', data=sales_train)
plt.show()


# we can clearly see that there are more records for Decembers of every year.
# I think it is because People tend to buy more items during the Chrismas.
#  Actually it is a seasonality. We can also find a trend that the entries are becoming fewer
# from 2013 to 2014.

sns.countplot(x='shop_id', data=sales_train)
plt.show()

# the shop 31 and 25 has the most bumber of entires, they may make a great contribution to the total sales trends.
# Therefore we need discover further for this two shops.

shop31=sales_train[sales_train['shop_id']==31]
sns.countplot(x='date_block_num', data=shop31)
plt.show()

# the trend for the shop 31 is similar for the total sales's trend, especially for the decembers of 2013 and 2014, it sold at least 10000 entries.
# The sales entries of this shop is above the average of every month.

sales_month = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum()
sales_month=pd.DataFrame(sales_month)
sales_month["date_block_num"]=sales_month.index

sns.barplot(x="date_block_num", y="item_cnt_day", data=sales_month , order=sales_month['date_block_num'])
plt.show()

# we can see that the total sales's trend is related to the record number of every month.But the number of item sold every month is much than the number of the record, since each item can be sold more than once. There is also a seasonality trend.
# For time series prediction, we need to remove the trend in the future.

dt = sales_train.iloc[:,:].values

revenue=np.multiply(dt[:,4:5],dt[:,5:])
re = pd.DataFrame(revenue)
re.columns = ['revenue']
new_salse=pd.concat([sales_train,re],axis=1)


sales_price = new_salse.groupby(['date_block_num'])['revenue'].sum()
sales_price=pd.DataFrame(sales_price)
sales_price['date']=sales_price.index.values
sns.barplot(x="date", y="revenue", data=sales_price , order=sales_price['date'])
plt.show()

# In the graph, the December of the 2013 and 2014 are also the most top month, but interesting thing is that even the item sold of 2013 is more than the 2014, the total revenue of 2014 is much than 2013.Inaddtion there isn't a decreasing trend for total revenue from 2013 to 2014.
# That is said, even the number of item sold decrease, the price of each item increase.

# 12/2013 sales suitation
price_2013=new_salse[new_salse['date_block_num']==11]
sales_day =price_2013.groupby(['date']).sum()
sales_day['dd']=sales_day.index.values
sns.barplot(x="dd", y="item_cnt_day", data=sales_day , order=sales_day['dd'] )
plt.xticks(rotation=90)
plt.show()

price_2014=new_salse[new_salse['date_block_num']==23]
sales_day14 =price_2014.groupby(['date']).sum()
sales_day14['dd']=sales_day.index.values
sns.barplot(x="dd", y="item_cnt_day", data=sales_day14 , order=sales_day14['dd'] )
plt.xticks(rotation=90)
plt.show()
