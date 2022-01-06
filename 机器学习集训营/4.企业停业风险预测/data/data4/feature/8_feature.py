
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 2.3.8 失信数据8breakfaith.csv提取特征
#     1. 企业失信的个数，BREAKFAITH_CNT
#     2. 企业第一个失信数据的日期，BREAKFAITH_FIRST_FIRST_TIME
#     3. 企业最后一个失信数据的日期，BREAKFAITH_FIRST_END_TIME
#     4. 企业第一个失信数据的到期日期，BREAKFAITH_END_FIRST_TIME
#     5. 企业第一个失信数据的日期与其结束日期的差值，BREAKFAITH_FIRST_END_FIRST_DIFF
# 
#     5. 企业第一个失信数据的日期与企业注册日期的差值，BREAKFAITH_FIRST_RGYEAR_DIFF
#     6. 企业最后一个失信数据的日期与企业注册日期的差值，BREAKFAITH_END_RGYEAR_DIFF
# 
#     7. 企业第一个失信数据的到期日期与企业注册日期的差值，BREAKFAITH_END_RGYEAR_DIFF
# 
#     8. 企业第一个失信数据的日期与企业第一次变更的差值，BREAKFAITH_FIRST_CHANGE_FIRST_DIFF
#     9. 企业最后一个失信数据的日期与企业第一次变更的差值，BREAKFAITH_END_CHANGE_FIRST_DIFF
#     
#     10. 企业第一个失信数据的日期与企业最后一次变更的差值，BREAKFAITH_FIRST_CHANGE_END_DIFF
#     11. 企业最后一个失信数据的日期与企业最后一次变更的差值，BREAKFAITH_END_CHANGE_END_DIFF
#     
#     ---------------------------------------------------------------------------------------
#     10. 企业平均每几个月的失信一次， BREAKFAITH_PRE_MONTH_CNT = BREAKFAITH_END_RGYEAR_DIFF / BREAKFAITH_CNT
#     
#     11. 企业失信的个数占所有失信个数个平均值的比例，BREAKFAITH_CNT_ALL_RATE
#     
#     12. 企业对应的大类HY的平均失信数据的个数，BREAKFAITH_HY_CNT_AVG
#     13. 企业对应大类HY的平均失信个数占所有失信平均个数的比例，BREAKFAITH_HY_CNT_ALL_RATE
#     14. 企业失信的个数占其对应的大类HY的失信平均值的比例，BREAKFAITH_CNT_HY_RATE
# 
#     ---------------------------------------------------------------------------------------
#     
#     12. 企业对应的大类HY的平均失信数据的个数，BREAKFAITH_ETYPE_CNT_AVG
#     13. 企业对应大类HY的平均失信个数占所有失信平均个数的比例，BREAKFAITH_ETYPE_CNT_ALL_RATE
#     14. 企业失信的个数占其对应的大类HY的失信平均值的比例，BREAKFAITH_CNT_ETYPE_RATE
# 
#     对应的MAX数据特征
#     
#     复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     BREAKFAITH_K_OPEN_CNT,
# 
#     
#     

# In[2]:

df_all = pd.read_csv("../data/alldata/df_data1234567.csv")
df_breakfaith = pd.read_csv("../data/public/8breakfaith.csv")


# In[3]:

# df_all.info()
# df_all.head()


# In[4]:


# df_breakfaith.info()
# df_breakfaith.head()


# In[5]:

# df_breakfaith['SXENDDATE'].value_counts()


# In[6]:

df_breakfaith['FBDATE'] = df_breakfaith['FBDATE'].map(lambda x:x.replace('年','-').replace('月',''))

df_breakfaith.loc[df_breakfaith['SXENDDATE'].isnull(),['SXENDDATE']] = '2018/1/1'
df_breakfaith.loc[df_breakfaith['SXENDDATE']=='2018/1/1',['SXENDDATE']] = '2018-01'


def time(x):
    y = x[:x.find('/')]
    m = int(x[x.find('/')+1:x.rfind('/')])
    if m < 10: m = '0'+str(m)
    else: m = str(m)
    return y + '-' + m

# df_breakfaith['FBDATE'] = df_breakfaith['FBDATE'].apply(time)
# df_breakfaith['SXENDDATE'] = df_breakfaith['SXENDDATE'].apply(time)


# In[7]:

df_breakfaith = df_breakfaith.sort_values(['FBDATE','SXENDDATE'])

# df_breakfaith.info()
# df_breakfaith.head()


# In[8]:

EIDS = set(df_breakfaith['EID'])

# print(len(EIDS))

columns = df_breakfaith.columns
df_xbreakfaith = pd.DataFrame(columns=columns)

# print(columns)


# In[9]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_breakfaith[df_breakfaith['EID'] == EID]
    row = [EID,tmp['TYPECODE'].values,tmp['FBDATE'].values,tmp['SXENDDATE'].values]
    
    df_xbreakfaith = df_xbreakfaith.append(pd.Series(row,columns),ignore_index=True)
    


# In[10]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xbreakfaith['BREAKFAITH_1_OPEN_CNT'] = df_xbreakfaith['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xbreakfaith['BREAKFAITH_2_OPEN_CNT'] = df_xbreakfaith['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xbreakfaith['BREAKFAITH_3_OPEN_CNT'] = df_xbreakfaith['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xbreakfaith['BREAKFAITH_5_OPEN_CNT'] = df_xbreakfaith['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))


# 文件8的数据量特别小，对特征几乎没有意思，这里添加一个时候存在 失信数据的特征 IS_BREAKFAITH
df_xbreakfaith['IS_BREAKFAITH'] = 1



# In[ ]:




# In[11]:

# df_xbreakfaith.info()
# df_xbreakfaith.head()


# In[12]:

df_xbreakfaith['BREAKFAITH_CNT'] = df_xbreakfaith['TYPECODE'].apply(lambda x: len(x))

df_xbreakfaith['BREAKFAITH_FIRST_FIRST_TIME'] = df_xbreakfaith['FBDATE'].apply(lambda x: x[0])
df_xbreakfaith['BREAKFAITH_FIRST_END_TIME'] = df_xbreakfaith['FBDATE'].apply(lambda x: x[-1])
df_xbreakfaith['BREAKFAITH_END_FIRST_TIME'] = df_xbreakfaith['SXENDDATE'].apply(lambda x: x[0])



# In[13]:

df_xbreakfaith.to_csv('../data/public/8breakfaith_1.csv',index=False,index_label=False)
df_xbreakfaith.columns



# In[14]:

df_all = pd.merge(df_all,df_xbreakfaith[['EID','BREAKFAITH_CNT','BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME',
                                       'BREAKFAITH_END_FIRST_TIME','BREAKFAITH_1_OPEN_CNT','BREAKFAITH_2_OPEN_CNT',
                                         'BREAKFAITH_3_OPEN_CNT','BREAKFAITH_5_OPEN_CNT','IS_BREAKFAITH']],how='left',on=['EID'])


# In[15]:

# df_all.info()
# df_all.head()


# In[16]:

# 所有有管时间的特征，用注册日期填充，其他的用0填充
df_all['BREAKFAITH_CNT'] = df_all['BREAKFAITH_CNT'].fillna(0)

# 不存在失信数据的填写0
df_all['IS_BREAKFAITH'] = df_all['IS_BREAKFAITH'].fillna(0)

modelist = [ 'BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME','BREAKFAITH_END_FIRST_TIME']

for d in modelist:
    df_all.loc[df_all[d].isnull(),d] = df_all.loc[df_all[d].isnull(),'RGYEAR']



# In[ ]:




# In[17]:

df_all[['BREAKFAITH_1_OPEN_CNT','BREAKFAITH_2_OPEN_CNT','BREAKFAITH_3_OPEN_CNT','BREAKFAITH_5_OPEN_CNT']] = df_all[['BREAKFAITH_1_OPEN_CNT','BREAKFAITH_2_OPEN_CNT','BREAKFAITH_3_OPEN_CNT','BREAKFAITH_5_OPEN_CNT']].fillna(0)


# In[18]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m


df_all['BREAKFAITH_FIRST_END_FIRST_DIFF'] = (df_all['BREAKFAITH_END_FIRST_TIME'] + ':' + df_all['BREAKFAITH_FIRST_FIRST_TIME']).apply(timeDiff)
df_all['BREAKFAITH_FIRST_RGYEAR_DIFF'] = (df_all['BREAKFAITH_FIRST_FIRST_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['BREAKFAITH_END_RGYEAR_DIFF'] = (df_all['BREAKFAITH_FIRST_END_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['BREAKFAITH_END_RGYEAR_DIFF'] = (df_all['BREAKFAITH_END_FIRST_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)


df_all['BREAKFAITH_FIRST_CHANGE_FIRST_DIFF'] = (df_all['BREAKFAITH_FIRST_FIRST_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['BREAKFAITH_END_CHANGE_FIRST_DIFF'] = (df_all['BREAKFAITH_FIRST_END_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['BREAKFAITH_FIRST_CHANGE_END_DIFF'] = (df_all['BREAKFAITH_FIRST_FIRST_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)
df_all['BREAKFAITH_END_CHANGE_END_DIFF'] = (df_all['BREAKFAITH_FIRST_END_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)


    # ---------------------------------------------------------------------------------------
    # 10. 企业平均每几个月的失信一次， BREAKFAITH_PRE_MONTH_CNT = BREAKFAITH_END_RGYEAR_DIFF / BREAKFAITH_CNT
    
    # 11. 企业失信的个数占所有失信个数个平均值的比例，BREAKFAITH_CNT_ALL_RATE
    
    # 12. 企业对应的大类HY的平均失信数据的个数，BREAKFAITH_HY_CNT_AVG
    # 13. 企业对应大类HY的平均失信个数占所有失信平均个数的比例，BREAKFAITH_HY_CNT_ALL_RATE
    # 14. 企业失信的个数占其对应的大类HY的失信平均值的比例，BREAKFAITH_CNT_HY_RATE

# In[19]:

df_all['BREAKFAITH_PRE_MONTH_CNT'] = df_all['BREAKFAITH_END_RGYEAR_DIFF'] / df_all['BREAKFAITH_CNT']

df_all['BREAKFAITH_CNT_ALL_RATE'] = df_all['BREAKFAITH_CNT'] / df_all['BREAKFAITH_CNT'].mean()

df_all['BREAKFAITH_CNT_ALL_RATE_MAX'] = df_all['BREAKFAITH_CNT'] / df_all['BREAKFAITH_CNT'].max()




# In[20]:

tmp = pd.DataFrame()

tmp['BREAKFAITH_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['BREAKFAITH_CNT'].mean()
tmp['BREAKFAITH_HY_CNT_ALL_RATE'] = tmp['BREAKFAITH_HY_CNT_AVG'] / df_all['BREAKFAITH_CNT'].mean()

tmp['BREAKFAITH_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['BREAKFAITH_CNT'].max()
tmp['BREAKFAITH_HY_CNT_ALL_RATE_MAX'] = tmp['BREAKFAITH_HY_CNT_MAX'] / df_all['BREAKFAITH_CNT'].max()


tmp['HY'] = tmp.index



# In[21]:

tmp1 = pd.DataFrame()

tmp1['BREAKFAITH_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BREAKFAITH_CNT'].mean()
tmp1['BREAKFAITH_ETYPE_CNT_ALL_RATE'] = tmp1['BREAKFAITH_ETYPE_CNT_AVG'] / df_all['BREAKFAITH_CNT'].mean()

tmp1['BREAKFAITH_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BREAKFAITH_CNT'].max()
tmp1['BREAKFAITH_ETYPE_CNT_ALL_RATE_MAX'] = tmp1['BREAKFAITH_ETYPE_CNT_MAX'] / df_all['BREAKFAITH_CNT'].max()


tmp1['ETYPE'] = tmp1.index



# In[22]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])


# In[23]:

df_all['BREAKFAITH_CNT_HY_RATE'] = df_all['BREAKFAITH_CNT'] / df_all['BREAKFAITH_HY_CNT_AVG']
df_all['BREAKFAITH_CNT_HY_RATE_MAX'] = df_all['BREAKFAITH_CNT'] / df_all['BREAKFAITH_HY_CNT_MAX']

df_all['BREAKFAITH_CNT_ETYPE_RATE'] = df_all['BREAKFAITH_CNT'] / df_all['BREAKFAITH_ETYPE_CNT_AVG']
df_all['BREAKFAITH_CNT_ETYPE_RATE_MAX'] = df_all['BREAKFAITH_CNT'] / df_all['BREAKFAITH_ETYPE_CNT_MAX']



# In[24]:

# df_all.info()
# df_all.head()


# In[25]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data12345678.csv',index=False,index_label=False)



# In[26]:

# df_all[['EID','IS_BREAKFAITH']].to_csv('../data/df_breakfaith_insert.csv',index=False,index_label=False)


# In[ ]:



