
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# QUALIFICATION -->QUALIFICATION
# qualification --> qualification
# ADDTYPE --> ADDTYPE
# BEGINDATE --> BEGINDATE
# EXPIRYDATE -->EXPIRYDATE

# # 2.3.8 失信数据10qualification.csv提取特征
#     1. 企业失信的个数，QUALIFICATION_CNT
#     2. 企业第一个失信数据的日期，QUALIFICATION_FIRST_FIRST_TIME
#     3. 企业最后一个失信数据的日期，QUALIFICATION_FIRST_END_TIME
#     4. 企业第一个失信数据的到期日期，QUALIFICATION_END_FIRST_TIME
#     5. 企业第一个失信数据的日期与其结束日期的差值，QUALIFICATION_FIRST_END_FIRST_DIFF
# 
#     5. 企业第一个失信数据的日期与企业注册日期的差值，QUALIFICATION_FIRST_RGYEAR_DIFF
#     6. 企业最后一个失信数据的日期与企业注册日期的差值，QUALIFICATION_END_RGYEAR_DIFF
# 
#     7. 企业第一个失信数据的到期日期与企业注册日期的差值，QUALIFICATION_END_RGYEAR_DIFF
# 
#     8. 企业第一个失信数据的日期与企业第一次变更的差值，QUALIFICATION_FIRST_CHANGE_FIRST_DIFF
#     9. 企业最后一个失信数据的日期与企业第一次变更的差值，QUALIFICATION_END_CHANGE_FIRST_DIFF
#     
#     10. 企业第一个失信数据的日期与企业最后一次变更的差值，QUALIFICATION_FIRST_CHANGE_END_DIFF
#     11. 企业最后一个失信数据的日期与企业最后一次变更的差值，QUALIFICATION_END_CHANGE_END_DIFF
#     
#     ---------------------------------------------------------------------------------------
#     10. 企业平均每几个月的失信一次， QUALIFICATION_PRE_MONTH_CNT = QUALIFICATION_END_RGYEAR_DIFF / QUALIFICATION_CNT
#     
#     11. 企业失信的个数占所有失信个数个平均值的比例，QUALIFICATION_CNT_ALL_RATE
#     
#     12. 企业对应的大类HY的平均失信数据的个数，QUALIFICATION_HY_CNT_AVG
#     13. 企业对应大类HY的平均失信个数占所有失信平均个数的比例，QUALIFICATION_HY_CNT_ALL_RATE
#     14. 企业失信的个数占其对应的大类HY的失信平均值的比例，QUALIFICATION_CNT_HY_RATE
# 
#     ---------------------------------------------------------------------------------------
#     
#     12. 企业对应的大类HY的平均失信数据的个数，QUALIFICATION_ETYPE_CNT_AVG
#     13. 企业对应大类HY的平均失信个数占所有失信平均个数的比例，QUALIFICATION_ETYPE_CNT_ALL_RATE
#     14. 企业失信的个数占其对应的大类HY的失信平均值的比例，QUALIFICATION_CNT_ETYPE_RATE
# 
#     对应的MAX数据特征
#     复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     QUALIFICATION_K_OPEN_CNT, QUALIFICATION_K_CLOSE_CNT,
#     

# In[2]:

df_all = pd.read_csv("../data/alldata/df_data123456789.csv")
df_qualification = pd.read_csv("../data/public/10qualification.csv",encoding='gbk')


# In[3]:

# df_all.info()
# df_all.head()


# In[4]:


# df_qualification.info()
# df_qualification.head()


# In[5]:

# df_qualification['EXPIRYDATE'].value_counts()


# In[6]:

df_qualification['BEGINDATE'] = df_qualification['BEGINDATE'].map(lambda x:x.replace('年','-').replace('月',''))
df_qualification.loc[df_qualification['EXPIRYDATE'].isnull(),['EXPIRYDATE']] = '2018年12月'

df_qualification['EXPIRYDATE'] = df_qualification['EXPIRYDATE'].map(lambda x:x.replace('年','-').replace('月',''))


def time(x):
    y = x[:x.find('/')]
    m = int(x[x.find('/')+1:x.rfind('/')])
    if m < 10: m = '0'+str(m)
    else: m = str(m)
        
    return y + '-' + m

# df_qualification['BEGINDATE'] = df_qualification['BEGINDATE'].apply(time)
# df_qualification['EXPIRYDATE'] = df_qualification['EXPIRYDATE'].apply(time)


# In[7]:

df_qualification = df_qualification.sort_values(['BEGINDATE','EXPIRYDATE'])

# df_qualification.info()
# df_qualification.head()


# In[8]:

EIDS = set(df_qualification['EID'])

# print(len(EIDS))

columns = df_qualification.columns
df_xqualification = pd.DataFrame(columns=columns)

# print(columns)


# In[9]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_qualification[df_qualification['EID'] == EID]
    row = [EID,tmp['ADDTYPE'].values,tmp['BEGINDATE'].values,tmp['EXPIRYDATE'].values]
    
    df_xqualification = df_xqualification.append(pd.Series(row,columns),ignore_index=True)
    


# In[10]:

# df_xqualification.info()
# df_xqualification.head()


# In[11]:

df_xqualification['QUALIFICATION_CNT'] = df_xqualification['ADDTYPE'].apply(lambda x: len(x))

df_xqualification['QUALIFICATION_FIRST_FIRST_TIME'] = df_xqualification['BEGINDATE'].apply(lambda x: x[0])
df_xqualification['QUALIFICATION_FIRST_END_TIME'] = df_xqualification['BEGINDATE'].apply(lambda x: x[-1])
df_xqualification['QUALIFICATION_END_FIRST_TIME'] = df_xqualification['EXPIRYDATE'].apply(lambda x: x[0])



# In[12]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xqualification['QUALIFICATION_1_OPEN_CNT'] = df_xqualification['BEGINDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xqualification['QUALIFICATION_2_OPEN_CNT'] = df_xqualification['BEGINDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xqualification['QUALIFICATION_3_OPEN_CNT'] = df_xqualification['BEGINDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xqualification['QUALIFICATION_5_OPEN_CNT'] = df_xqualification['BEGINDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))

df_xqualification['QUALIFICATION_1_CLOSE_CNT'] = df_xqualification['EXPIRYDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xqualification['QUALIFICATION_2_CLOSE_CNT'] = df_xqualification['EXPIRYDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xqualification['QUALIFICATION_3_CLOSE_CNT'] = df_xqualification['EXPIRYDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xqualification['QUALIFICATION_5_CLOSE_CNT'] = df_xqualification['EXPIRYDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))


# 是否有QUALIFICATION数据，特征 IS_QUALIFICATION

df_xqualification['IS_QUALIFICATION'] = 1


# In[13]:

df_xqualification.to_csv('../data/public/10qualification_1.csv',index=False,index_label=False)
df_xqualification.columns



# In[14]:

df_all = pd.merge(df_all,df_xqualification[['EID','QUALIFICATION_CNT','QUALIFICATION_FIRST_FIRST_TIME', 'QUALIFICATION_FIRST_END_TIME',
                                           'QUALIFICATION_END_FIRST_TIME','QUALIFICATION_1_OPEN_CNT','QUALIFICATION_2_OPEN_CNT',
                                            'QUALIFICATION_3_OPEN_CNT','QUALIFICATION_5_OPEN_CNT','QUALIFICATION_1_CLOSE_CNT',
                                            'QUALIFICATION_2_CLOSE_CNT','QUALIFICATION_3_CLOSE_CNT','QUALIFICATION_5_CLOSE_CNT',
                                           'IS_QUALIFICATION']],how='left',on=['EID'])


# In[15]:

# df_all.info()
# df_all.head()


# In[16]:

# 所有有管时间的特征，用注册日期填充，其他的用0填充
df_all['QUALIFICATION_CNT'] = df_all['QUALIFICATION_CNT'].fillna(0)

df_all['IS_QUALIFICATION'] = df_all['IS_QUALIFICATION'].fillna(0)

modelist = [ 'QUALIFICATION_FIRST_FIRST_TIME', 'QUALIFICATION_FIRST_END_TIME','QUALIFICATION_END_FIRST_TIME']

for d in modelist:
    df_all.loc[df_all[d].isnull(),d] = df_all.loc[df_all[d].isnull(),'RGYEAR']



# In[17]:

df_all[['QUALIFICATION_1_OPEN_CNT','QUALIFICATION_2_OPEN_CNT','QUALIFICATION_3_OPEN_CNT','QUALIFICATION_5_OPEN_CNT']] = df_all[['QUALIFICATION_1_OPEN_CNT','QUALIFICATION_2_OPEN_CNT','QUALIFICATION_3_OPEN_CNT','QUALIFICATION_5_OPEN_CNT']].fillna(0)


df_all[['QUALIFICATION_1_CLOSE_CNT','QUALIFICATION_2_CLOSE_CNT','QUALIFICATION_3_CLOSE_CNT','QUALIFICATION_5_CLOSE_CNT']] = df_all[['QUALIFICATION_1_CLOSE_CNT','QUALIFICATION_2_CLOSE_CNT','QUALIFICATION_3_CLOSE_CNT','QUALIFICATION_5_CLOSE_CNT']].fillna(0)


# In[18]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m


df_all['QUALIFICATION_FIRST_END_FIRST_DIFF'] = (df_all['QUALIFICATION_END_FIRST_TIME'] + ':' + df_all['QUALIFICATION_FIRST_FIRST_TIME']).apply(timeDiff)
df_all['QUALIFICATION_FIRST_RGYEAR_DIFF'] = (df_all['QUALIFICATION_FIRST_FIRST_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['QUALIFICATION_END_RGYEAR_DIFF'] = (df_all['QUALIFICATION_FIRST_END_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['QUALIFICATION_END_RGYEAR_DIFF'] = (df_all['QUALIFICATION_END_FIRST_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)


df_all['QUALIFICATION_FIRST_CHANGE_FIRST_DIFF'] = (df_all['QUALIFICATION_FIRST_FIRST_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['QUALIFICATION_END_CHANGE_FIRST_DIFF'] = (df_all['QUALIFICATION_FIRST_END_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['QUALIFICATION_FIRST_CHANGE_END_DIFF'] = (df_all['QUALIFICATION_FIRST_FIRST_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)
df_all['QUALIFICATION_END_CHANGE_END_DIFF'] = (df_all['QUALIFICATION_FIRST_END_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)


    # ---------------------------------------------------------------------------------------
    # 10. 企业平均每几个月的失信一次， QUALIFICATION_PRE_MONTH_CNT = QUALIFICATION_END_RGYEAR_DIFF / QUALIFICATION_CNT
    
    # 11. 企业失信的个数占所有失信个数个平均值的比例，QUALIFICATION_CNT_ALL_RATE
    
    # 12. 企业对应的大类HY的平均失信数据的个数，QUALIFICATION_HY_CNT_AVG
    # 13. 企业对应大类HY的平均失信个数占所有失信平均个数的比例，QUALIFICATION_HY_CNT_ALL_RATE
    # 14. 企业失信的个数占其对应的大类HY的失信平均值的比例，QUALIFICATION_CNT_HY_RATE

# In[19]:

df_all['QUALIFICATION_PRE_MONTH_CNT'] = df_all['QUALIFICATION_END_RGYEAR_DIFF'] / df_all['QUALIFICATION_CNT']

df_all['QUALIFICATION_CNT_ALL_RATE'] = df_all['QUALIFICATION_CNT'] / df_all['QUALIFICATION_CNT'].mean()

df_all['QUALIFICATION_CNT_ALL_RATE_MAX'] = df_all['QUALIFICATION_CNT'] / df_all['QUALIFICATION_CNT'].max()




# In[20]:

tmp = pd.DataFrame()

tmp['QUALIFICATION_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['QUALIFICATION_CNT'].mean()
tmp['QUALIFICATION_HY_CNT_ALL_RATE'] = tmp['QUALIFICATION_HY_CNT_AVG'] / df_all['QUALIFICATION_CNT'].mean()

tmp['QUALIFICATION_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['QUALIFICATION_CNT'].max()
tmp['QUALIFICATION_HY_CNT_ALL_RATE_MAX'] = tmp['QUALIFICATION_HY_CNT_MAX'] / df_all['QUALIFICATION_CNT'].max()


tmp['HY'] = tmp.index



# In[21]:

tmp1 = pd.DataFrame()

tmp1['QUALIFICATION_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['QUALIFICATION_CNT'].mean()
tmp1['QUALIFICATION_ETYPE_CNT_ALL_RATE'] = tmp1['QUALIFICATION_ETYPE_CNT_AVG'] / df_all['QUALIFICATION_CNT'].mean()

tmp1['QUALIFICATION_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['QUALIFICATION_CNT'].max()
tmp1['QUALIFICATION_ETYPE_CNT_ALL_RATE_MAX'] = tmp1['QUALIFICATION_ETYPE_CNT_MAX'] / df_all['QUALIFICATION_CNT'].max()


tmp1['ETYPE'] = tmp1.index



# In[22]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])


# In[23]:

df_all['QUALIFICATION_CNT_HY_RATE'] = df_all['QUALIFICATION_CNT'] / df_all['QUALIFICATION_HY_CNT_AVG']
df_all['QUALIFICATION_CNT_HY_RATE_MAX'] = df_all['QUALIFICATION_CNT'] / df_all['QUALIFICATION_HY_CNT_MAX']

df_all['QUALIFICATION_CNT_ETYPE_RATE'] = df_all['QUALIFICATION_CNT'] / df_all['QUALIFICATION_ETYPE_CNT_AVG']
df_all['QUALIFICATION_CNT_ETYPE_RATE_MAX'] = df_all['QUALIFICATION_CNT'] / df_all['QUALIFICATION_ETYPE_CNT_MAX']



# In[24]:

# df_all.info()
# df_all.head()


# In[25]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data1234567890.csv',index=False,index_label=False)



# In[ ]:



