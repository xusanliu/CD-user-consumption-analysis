#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


columns=['user_id','order_dt','order_products','order_amount']
df=pd.read_table('CDNOW_master.txt',names=columns,sep='\s+')


# In[3]:


df.head()


# In[4]:


df.info()


# 发现日期栏是有问题的，数据类型为int，应该为日期形式

# In[5]:


df.describe()


# In[6]:


df['order_dt']=pd.to_datetime(df.order_dt,format="%Y%m%d")
df.head()


# In[7]:


#提取日期中的月份
df['month']=df.order_dt.values.astype('datetime64[M]')
df.head()


# # 一、用户消费趋势分析（按月）
# -每月的消费金额  
# -每月的消费次数  
# -每月购买产品的数量  
# -每月消费人数  
# -每月用户平均消费金额  
# -每月用户平均消费次数的趋势

# In[8]:


#每月的消费金额
grouped_month=df.groupby('month')
order_month_amount=grouped_month.order_amount.sum()
order_month_amount.plot()


# 消费金额在1-3月处于高位，并且呈上升趋势，随后经历大幅下降，逐渐趋于平稳波动状态，在9月处于最低谷

# In[9]:


#每月消费的次数
order_num=grouped_month.user_id.count()
order_num.plot()


# 消费次数在1-3月处于高水平，均超过10000，最高点在14000 随后经历下降，逐渐趋于平稳波动状态7-12月均低于3000

# In[10]:


#每月购买产品的数量
product_num=grouped_month.order_products.sum()
product_num.plot()


# 每月购买产品的数量在1-3月呈上升趋势，在3月到达峰值之后开始急剧下降，4月态势逐渐平稳，最低值在8-9月

# In[11]:


#每月消费人数
user_num=grouped_month.user_id.apply(lambda x:len((x.drop_duplicates())))
user_num.plot()


# 用户数在2-3月水平稳定，保持在10000以上 3月开始呈急剧下降趋势，到4月趋于平稳4-12月的用户数低于4000 8-9月处于最低谷 约为2000

# In[12]:


#用数据透视的方法
df.pivot_table(index='month',
               values=['order_amount','order_products','user_id'],
               aggfunc={
                   'order_amount':'sum',
                   'order_products':'sum',
                   'user_id':'count'
               }
)


# In[13]:


#每月用户平均消费金额
avg_cost=order_month_amount/user_num
avg_cost.plot()


# In[14]:


#每月用户平均消费次数的趋势
avg_num=order_num/user_num
avg_num.plot()


# # 二、用户个体消费分析
# -用户消费金额、消费次数的描点统计  
# -用户金额和消费的散点图  
# -用户消费金额的分布图  
# -用户消费次数的分布图  
# -用户累计消费金额百分比（百分之多少的用户消费了百分之多少的金额）

# In[15]:


grouped_user=df.groupby('user_id')


# In[16]:


grouped_user.sum().describe()


# 用户购买数目的平均值是7 但是中位数却是3 说明少部分的用户购买了大量的产品  
# 购买金额的均值是106 但中位数只有43 说明少部分的用户花费占了大部分

# In[17]:


grouped_user.sum().query('order_amount<2000').plot.scatter(x='order_amount',y='order_products')


# In[18]:


grouped_user.sum().query('order_products<100').order_products.plot.hist(bins=20)


# In[19]:


user_consum=grouped_user.sum().sort_values('order_amount').apply(lambda x:x.cumsum()/x.sum())
user_consum


# In[20]:


user_consum.reset_index()


# In[21]:


user_consum.reset_index().order_amount.plot()


# 由图可知，50%的用户贡献了约15%的消费额

# 横轴为用户数

# # 三、用户消费行为
# -用户第一次消费（首购）  
# -用户最后一次消费  
# -新老客户消费比  
#   -多少用户仅消费了一次  
#   -每月新客占比  
# -用户分层  
#   -RFM
#   -新、老、活跃、回流、流失  
# -用户购买周期（按订单）  
#   -用户消费周期描述  
#   -用户消费周期分布  
# -用户生命周期（按第一次，最后一次消费）  
#   -用户生命周期描述  
#   -用户生命周期分布

# In[22]:


grouped_user.min().order_dt.value_counts().plot()


# 用户第一次购买（首购）的分布，集中在前三个月

# In[23]:


grouped_user.max().order_dt.value_counts().plot()


# 大部分的最后一次购买集中在前三个月，而用户的首次购买也集中在前三个月，说明有相当一部分的用户首次购买之后便不再进行下一次的购买了

# In[24]:


user_life=grouped_user.order_dt.agg(['max','min'])
user_life.head()


# In[25]:


(user_life['max']==user_life['min']).value_counts()


# 有超过一半的用户，只消费了一次

# In[26]:


rfm=df.pivot_table(index='user_id',
                  values=['order_products','order_amount','order_dt'],
                   aggfunc={
                       'order_dt':'max',
                       'order_amount':'sum',
                       'order_products':'sum'
                   }
                  )
rfm.head()


# In[27]:


rfm['R']=-(rfm.order_dt-rfm.order_dt.max())/np.timedelta64(1,'D')
rfm['R']


# In[28]:


rfm.rename(columns={'order_products':'F','order_amount':'M'},inplace=True)


# In[29]:


rfm


# In[30]:


def rfm_func(x):
    level=x.apply(lambda x:'1' if x>=0 else '0')
    label=level.R+level.F+level.M
    d={
        '111':'重要价值客户',
        '011':'重要保持客户',
        '101':'重要发展客户',
        '001':'重要挽留客户',
        '110':'一般价值客户',
        '010':'一般保持客户',
        '100':'一般发展客户',
        '000':'一般挽留客户'
    }
    result=d[label]
    return result
rfm['label']=rfm[['R','F','M']].apply(lambda x:x-x.mean()).apply(rfm_func,axis=1)


# In[31]:


rfm


# In[32]:


rfm.groupby('label').sum()


# 重要保持客户的消费总金额最大，消费频次数也最高

# In[33]:


rfm.groupby('label').count()


# 一般发展客户的人数最多，消费金额和消费频次最高的重要保持客户人数为4554

# In[34]:


rfm.loc[rfm.label=='重要价值客户','color']='g'
rfm.loc[~(rfm.label=='重要价值客户'),'color']='r'
rfm.plot.scatter('F','R',c=rfm.color)


# In[35]:


pivot_counts=df.pivot_table(
                        index='user_id',
    columns='month',
    values='order_dt',
    aggfunc='count'
).fillna(0)
pivot_counts.head()


# In[36]:


df_purchase=pivot_counts.applymap(lambda x:1 if x>0 else 0)
df_purchase.head()


# In[37]:


def active_status(data):
    status = []
    for i in range(18):
        #若本月没有消费
        if data[i]==0:
            if len(status)>0:
                if status[i-1]=='unreg':
                    status.append('unreg')
                else:
                    status.append('unactive')
            else:
                status.append('unreg')
        #若本月消费
        else:
            if len(status)==0:
                status.append('new')
            else:
                if status[i-1]=='unactive':
                    status.append('return')
                elif status[i-1]=='unreg':
                    status.append('new')
                else:
                    status.append('active')
    return status                   


# In[38]:


purchase_status=df_purchase.apply(active_status,axis=1)


# In[39]:


purchase_status


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




