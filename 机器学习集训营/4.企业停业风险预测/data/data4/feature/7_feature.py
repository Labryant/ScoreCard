
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 被执行数据7lawsuit.csv提取特征
#     1. 企业被执行的案件个数，LAWSUIT_CNT
#     2. 企业被执行的案件得金额的均值，LAWSUIT_LAWAMOUNT_MEAN
#     3. 企业被执行的案件得金额的和，LAWSUIT_LAWAMOUNT_SUM
#     4. 企业第一被执行的时间， LAWSUIT_FIRST_TIME
#     5. 企业最后一被执行的时间， LAWSUIT_END_TIME
#     6. 企业第一个被执行的时间与企业注册的时间的差值， LAWSUIT_FIRST_RGYEAR_TIME_DIFF
#     7. 企业最后一个被执行的时间与企业注册的时间的差值， LAWSUIT_END_RGYEAR_TIME_DIFF
#     8. 企业第一个被执行的时间与企业第一次变更时间的差值， LAWSUIT_FIRST_CHANGE_FIRST_TIME_DIFF
#     9. 企业第一个被执行的时间与企业最后一次变更时间的差值， LAWSUIT_FIRST_CHANGE_END_TIME_DIFF
#     
#     ---------------------------------------------------------------------------------------
#     10. 企业平均没几个月被执行的一次， LAWSUIT_PRE_MONTH_CNT = LAWSUIT_END_RGYEAR_TIME_DIFF / LAWSUIT_CNT
#     
#     11. 企业被执行的个数占所有执行个数个平均值的比例，LAWSUIT_CNT_ALL_RATE    
#     
#     12. 企业对应的大类HY的平均执行数据的个数，LAWSUIT_HY_CNT_AVG
#     13. 企业对应大类HY的平均执行个数占所有执行平均个数的比例，LAWSUIT_HY_CNT_ALL_RATE
#     14. 企业执行的个数占其对应的大类HY的执行平均值的比例，LAWSUIT_CNT_HY_RATE
# 
# 
#     15. 企业被执行的金额占所有执行金额个平均值的比例，LAWSUIT_LAWAMOUNT_MEAN_ALL_RATE
#     
#     16. 企业对应的大类HY的平均金额数据的数值，LAWSUIT_LAWAMOUNT_MEAN_HY_AVG
#     17. 企业对应大类HY的平均执行金额占所有执行平均金额的比例，LAWSUIT_LAWAMOUNT_MEAN_HY_ALL_RATE
#     18. 企业执行的金额占其对应的大类HY的执行金额平均值的比例，LAWSUIT_LAWAMOUNT_MEAN_HY_RATE
# 
#     ---------------------------------------------------------------------------------------
#     
#     12. 企业对应的大类HY的平均执行数据的个数，LAWSUIT_ETYPE_CNT_AVG
#     13. 企业对应大类HY的平均执行个数占所有执行平均个数的比例，LAWSUIT_ETYPE_CNT_ALL_RATE
#     14. 企业执行的个数占其对应的大类HY的执行平均值的比例，LAWSUIT_CNT_ETYPE_RATE
# 
#     
#     16. 企业对应的大类HY的平均金额数据的数值，LAWSUIT_LAWAMOUNT_MEAN_ETYPE_AVG
#     17. 企业对应大类HY的平均执行金额占所有执行平均金额的比例，LAWSUIT_LAWAMOUNT_MEAN_ETYPE_ALL_RATE
#     18. 企业执行的金额占其对应的大类HY的执行金额平均值的比例，LAWSUIT_LAWAMOUNT_MEAN_ETYPE_RATE
# 
#     对应的max数据的特征
#     
#      复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     LAWSUIT_K_OPEN_CNT,
# 

# In[2]:

df_all = pd.read_csv("../data/alldata/df_data123456.csv")
df_lawsuit = pd.read_csv("../data/public/7lawsuit.csv")


# In[3]:

# df_all.info()
# df_all.head()


# In[4]:

# df_lawsuit.info()
# df_lawsuit.head()


# In[5]:

df_lawsuit['LAWDATE'] = df_lawsuit['LAWDATE'].map(lambda x:x.replace('年','-').replace('月',''))


# In[6]:

df_lawsuit = df_lawsuit.sort_values(['LAWDATE'])

EIDS = set(df_lawsuit['EID'])

# print(len(EIDS))

columns = df_lawsuit.columns
df_xlawsuit = pd.DataFrame(columns=columns)

# print(columns)


# In[7]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_lawsuit[df_lawsuit['EID'] == EID]
    row = [EID,tmp['TYPECODE'].values,tmp['LAWDATE'].values,tmp['LAWAMOUNT'].values]
    
    df_xlawsuit = df_xlawsuit.append(pd.Series(row,columns),ignore_index=True)
    


# In[8]:

# df_xlawsuit.info()
# df_xlawsuit.head()


# In[9]:

df_xlawsuit['LAWSUIT_CNT'] = df_xlawsuit['TYPECODE'].apply(lambda x: len(x))
df_xlawsuit['LAWSUIT_LAWAMOUNT_MEAN'] = df_xlawsuit['LAWAMOUNT'].apply(lambda x: np.mean(x))
df_xlawsuit['LAWSUIT_LAWAMOUNT_SUM'] = df_xlawsuit['LAWAMOUNT'].apply(lambda x: np.sum(x))


# In[10]:

df_xlawsuit['LAWSUIT_FIRST_TIME'] = df_xlawsuit['LAWDATE'].apply(lambda x: x[0][:7])
df_xlawsuit['LAWSUIT_END_TIME'] = df_xlawsuit['LAWDATE'].apply(lambda x: x[-1][:7])


# In[11]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xlawsuit['LAWSUIT_1_OPEN_CNT'] = df_xlawsuit['LAWDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xlawsuit['LAWSUIT_2_OPEN_CNT'] = df_xlawsuit['LAWDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xlawsuit['LAWSUIT_3_OPEN_CNT'] = df_xlawsuit['LAWDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xlawsuit['LAWSUIT_5_OPEN_CNT'] = df_xlawsuit['LAWDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))



# In[ ]:




# In[12]:

df_xlawsuit.to_csv('../data/public/7lawsuit_1.csv',index=False,index_label=False)
df_xlawsuit.columns



# In[13]:

df_all = pd.merge(df_all,df_xlawsuit[['EID', 'LAWSUIT_CNT','LAWSUIT_LAWAMOUNT_MEAN', 'LAWSUIT_LAWAMOUNT_SUM', 'LAWSUIT_FIRST_TIME',
                                       'LAWSUIT_END_TIME','LAWSUIT_1_OPEN_CNT','LAWSUIT_2_OPEN_CNT',
                                      'LAWSUIT_3_OPEN_CNT','LAWSUIT_5_OPEN_CNT']],how='left',on=['EID'])


# In[14]:

# df_all.info()
# df_all.head()


# In[15]:

# 所有的关于时间的空值用2018-01填充，假设这些企业被执行的数据发生在下一年，其他用0填充便是没有被执行
meanlist = ['LAWSUIT_FIRST_TIME','LAWSUIT_END_TIME']
modelist = ['LAWSUIT_CNT','LAWSUIT_LAWAMOUNT_MEAN', 'LAWSUIT_LAWAMOUNT_SUM']

for d in meanlist:
    df_all.loc[df_all[d].isnull(),d] = '2018-01'

for c in modelist:
    df_all.loc[df_all[c].isnull(),c] = 0


# In[16]:

df_all[['LAWSUIT_1_OPEN_CNT','LAWSUIT_2_OPEN_CNT','LAWSUIT_3_OPEN_CNT','LAWSUIT_5_OPEN_CNT']] = df_all[['LAWSUIT_1_OPEN_CNT','LAWSUIT_2_OPEN_CNT','LAWSUIT_3_OPEN_CNT','LAWSUIT_5_OPEN_CNT']].fillna(0)


# In[17]:

def timeDiff(x):
    x = x.replace('年','-')
    x = x.replace('月','')
    
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    
    
    
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m


df_all['LAWSUIT_FIRST_RGYEAR_TIME_DIFF'] = (df_all['LAWSUIT_FIRST_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['LAWSUIT_END_RGYEAR_TIME_DIFF'] = (df_all['LAWSUIT_END_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)

df_all['LAWSUIT_FIRST_CHANGE_FIRST_TIME_DIFF'] = (df_all['LAWSUIT_FIRST_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['LAWSUIT_FIRST_CHANGE_END_TIME_DIFF'] = (df_all['LAWSUIT_FIRST_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)

# ---------------------------------------------------------------------------------------
#     10. 企业平均没几个月被执行的一次， LAWSUIT_PRE_MONTH_CNT = LAWSUIT_END_RGYEAR_TIME_DIFF / LAWSUIT_CNT
    
#     11. 企业被执行的个数占所有执行个数个平均值的比例，LAWSUIT_CNT_ALL_RATE
    
#     12. 企业对应的大类HY的平均执行数据的个数，LAWSUIT_HY_CNT_AVG
#     13. 企业对应大类HY的平均执行个数占所有执行平均个数的比例，LAWSUIT_HY_CNT_ALL_RATE
#     14. 企业执行的个数占其对应的大类HY的执行平均值的比例，LAWSUIT_CNT_HY_RATE


#     15. 企业被执行的金额占所有执行金额个平均值的比例，LAWSUIT_LAWAMOUNT_MEAN_ALL_RATE
    
#     16. 企业对应的大类HY的平均金额数据的数值，LAWSUIT_LAWAMOUNT_MEAN_HY_AVG
#     17. 企业对应大类HY的平均执行金额占所有执行平均金额的比例，LAWSUIT_LAWAMOUNT_MEAN_ALL_RATE
#     18. 企业执行的金额占其对应的大类HY的执行金额平均值的比例，LAWSUIT_LAWAMOUNT_MEAN_HY_RATE
    
# In[18]:

df_all['LAWSUIT_PRE_MONTH_CNT'] = df_all['LAWSUIT_END_RGYEAR_TIME_DIFF'] / df_all['LAWSUIT_CNT']
df_all['LAWSUIT_CNT_ALL_RATE'] = df_all['LAWSUIT_CNT'] / df_all['LAWSUIT_CNT'].mean()

df_all['LAWSUIT_LAWAMOUNT_MEAN_ALL_RATE'] = df_all['LAWSUIT_LAWAMOUNT_MEAN'] / df_all['LAWSUIT_LAWAMOUNT_MEAN'].mean()


df_all['LAWSUIT_CNT_ALL_RATE_MAX'] = df_all['LAWSUIT_CNT'] / df_all['LAWSUIT_CNT'].max()

df_all['LAWSUIT_LAWAMOUNT_MEAN_ALL_RATE_MAX'] = df_all['LAWSUIT_LAWAMOUNT_MEAN'] / df_all['LAWSUIT_LAWAMOUNT_MEAN'].max()



# In[19]:

tmp = pd.DataFrame()

tmp['LAWSUIT_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['LAWSUIT_CNT'].mean()
tmp['LAWSUIT_HY_CNT_ALL_RATE'] = tmp['LAWSUIT_HY_CNT_AVG'] / df_all['LAWSUIT_CNT'].mean()

tmp['LAWSUIT_LAWAMOUNT_MEAN_HY_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['LAWSUIT_LAWAMOUNT_MEAN'].mean()
tmp['LAWSUIT_LAWAMOUNT_MEAN_HY_ALL_RATE'] = tmp['LAWSUIT_LAWAMOUNT_MEAN_HY_AVG'] / df_all['LAWSUIT_LAWAMOUNT_MEAN'].mean()


tmp['LAWSUIT_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['LAWSUIT_CNT'].max()
tmp['LAWSUIT_HY_CNT_ALL_RATE_MAX'] = tmp['LAWSUIT_HY_CNT_MAX'] / df_all['LAWSUIT_CNT'].max()

tmp['LAWSUIT_LAWAMOUNT_MEAN_HY_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['LAWSUIT_LAWAMOUNT_MEAN'].max()
tmp['LAWSUIT_LAWAMOUNT_MEAN_HY_ALL_RATE_MAX'] = tmp['LAWSUIT_LAWAMOUNT_MEAN_HY_MAX'] / df_all['LAWSUIT_LAWAMOUNT_MEAN'].max()


tmp['HY'] = tmp.index



# In[20]:

tmp1 = pd.DataFrame()

tmp1['LAWSUIT_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['LAWSUIT_CNT'].mean()
tmp1['LAWSUIT_ETYPE_CNT_ALL_RATE'] = tmp1['LAWSUIT_ETYPE_CNT_AVG'] / df_all['LAWSUIT_CNT'].mean()

tmp1['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['LAWSUIT_LAWAMOUNT_MEAN'].mean()
tmp1['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_ALL_RATE'] = tmp1['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_AVG'] / df_all['LAWSUIT_LAWAMOUNT_MEAN'].mean()


tmp1['LAWSUIT_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['LAWSUIT_CNT'].max()
tmp1['LAWSUIT_ETYPE_CNT_ALL_RATE_MAX'] = tmp1['LAWSUIT_ETYPE_CNT_MAX'] / df_all['LAWSUIT_CNT'].max()

tmp1['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['LAWSUIT_LAWAMOUNT_MEAN'].max()
tmp1['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_ALL_RATE_MAX'] = tmp1['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_MAX'] / df_all['LAWSUIT_LAWAMOUNT_MEAN'].max()


tmp1['ETYPE'] = tmp1.index




# In[21]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])



# In[22]:

df_all['LAWSUIT_CNT_HY_RATE'] = df_all['LAWSUIT_CNT'] / df_all['LAWSUIT_HY_CNT_AVG']
df_all['LAWSUIT_LAWAMOUNT_MEAN_HY_RATE'] = df_all['LAWSUIT_LAWAMOUNT_MEAN'] / df_all['LAWSUIT_LAWAMOUNT_MEAN_HY_AVG']

df_all['LAWSUIT_CNT_HY_RATE_MAX'] = df_all['LAWSUIT_CNT'] / df_all['LAWSUIT_HY_CNT_MAX']
df_all['LAWSUIT_LAWAMOUNT_MEAN_HY_RATE_MAX'] = df_all['LAWSUIT_LAWAMOUNT_MEAN'] / df_all['LAWSUIT_LAWAMOUNT_MEAN_HY_MAX']



# In[23]:

df_all['LAWSUIT_CNT_ETYPE_RATE'] = df_all['LAWSUIT_CNT'] / df_all['LAWSUIT_ETYPE_CNT_AVG']
df_all['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_RATE'] = df_all['LAWSUIT_LAWAMOUNT_MEAN'] / df_all['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_AVG']

df_all['LAWSUIT_CNT_ETYPE_RATE_MAX'] = df_all['LAWSUIT_CNT'] / df_all['LAWSUIT_ETYPE_CNT_MAX']
df_all['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_RATE_MAX'] = df_all['LAWSUIT_LAWAMOUNT_MEAN'] / df_all['LAWSUIT_LAWAMOUNT_MEAN_ETYPE_MAX']



# In[24]:

# df_all.info()
# df_all.head()


# In[25]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data1234567.csv',index=False,index_label=False)



# In[ ]:



