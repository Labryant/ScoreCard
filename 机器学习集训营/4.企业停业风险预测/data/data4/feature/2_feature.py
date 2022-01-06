
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# # 企业变更数据的对应特征(2alter.csv)
#     
#     1. 变更的事件的总个数, ALTERNO_SUM
#     2. 变更的事件的类型的种类数, ALTERNO_CNT
#     3. 变更的事件的类型的种类数 / 所有事件的总总的类别数, ALTERNO_CNT_RATE
#     4. 第一次变更时间, FIRST_CHANGE_TIME
#     5. 第一次变更距离企业注册时间的差（单位：月）,不存在变更事件的用0填充, FIRST_CHANGE_TIME_DIFF
#     6. 最后一次变更时间, END_CHANGE_TIME
#     7. 最后一次变更时间距离企业注册时间的差值，END_CHANGE_TIME_DIFF
#     8. 最后一次变更距离第一次事件变更时间的差（单位：月）,不存在变更事件的用0填充, END_FIRST_CHANGE_TIME_DIFF
# 
#     9. 变更的事件是否发生了前后的值得变化（1/0）, IS_CHANGE
#     10. 第一次变更事件的类型，CHANGE_FIRST_ALTERNO
#     11. 最后一次变更事件的类型，CHANGE_END_ALTERNO
#     12. 最多的变更事件的类型，CHANGE_ALTERNO_MUCNID
#     
#     新加特征
#     -------------------------------------------------------------------------------------------------------
#     13. 平均每几个月变更一次事件，CHANGE_PRE_CNT = ALTERNO_SUM / END_CHANGE_TIME_DIFF
#     14. 每个企业变更事件的个数占所有企业变更事件平均值的比例， CHANGE_CNT_RATE = ALTERNO_SUM / AVG 
#     15. 企业对应大类HY的平均变更次数，CHANGE_HY_CNT_AVG
#     16. 企业对应大类HY的平均变更次数占所有变更时间的个数的平均值的比例，CHANGE_HY_CNT_ALL_AVG
#     
#     17. 企业变更次数占其对应大类的平均次数的比例，CHANGE_CNT_HY_RATE = ALTERNO_SUM / CHANGE_HY_CNT_AVG
#     
#     新加特征1
#     -------------------------------------------------------------------------------------------------------
#     15. 企业对应大类ETYPE的平均变更次数，CHANGE_ETYPE_CNT_AVG
#     16. 企业对应大类ETYPE的平均变更次数占所有变更时间的个数的平均值的比例，CHANGE_ETYPE_CNT_ALL_AVG
#     17. 企业变更次数占其对应大类ETYPE的平均次数的比例，CHANGE_CNT_ETYPE_RATE = ALTERNO_SUM / CHANGE_ETYPE_CNT_AVG
#     
#     
#     添加对应的最大值的特征
#     
#     复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     CHANGE_K_CNT
# 
# 

# In[32]:

df_all = pd.read_csv("../data/alldata/df_data1.csv")
df_change = pd.read_csv("../data/public/2alter.csv")

# In[12]:

# df_all.info()
# df_all.head()


# In[13]:

# df_change.info()
# df_change.head()


# In[ ]:


# In[14]:

df_change = df_change.sort_values(['EID','ALTDATE'])


# In[15]:

df_change = df_change.fillna('0')
columns = ['EID', 'ALTERNO', 'ALTDATE', 'ALTBE', 'ALTAF']
df_xchange = pd.DataFrame(columns=['EID', 'ALTERNO', 'ALTDATE', 'ALTBE', 'ALTAF'])

# In[16]:

EIDS = set(df_change['EID'])

# len(EIDS)

# In[17]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_change[df_change['EID'] == EID]
    row = [EID,tmp['ALTERNO'].values,tmp['ALTDATE'].values,tmp['ALTBE'].values,tmp['ALTAF'].values]
    
    df_xchange = df_xchange.append(pd.Series(row,columns),ignore_index=True)
    


# In[18]:

# df_xchange.info()
# df_xchange.head()

def getFAlterno(x):
    m = x[0]
    if m == 'A_015':
        m = m[3:]
    return m

def getEAlterno(x):
    m = x[-1]
    if m == 'A_015':
        m = m[3:]
    return m

def getMAlterno(x):
    x = list(x)
    m = x[0]
    k = 0
    for i in x:
        if k<x.count(i):
            k=x.count(i)
            m = i
    if m == 'A_015':
        m = m[3:]
    return m


df_xchange['CHANGE_FIRST_ALTERNO'] = df_xchange['ALTERNO'].apply(getFAlterno)
df_xchange['CHANGE_END_ALTERNO'] = df_xchange['ALTERNO'].apply(getEAlterno)
df_xchange['CHANGE_ALTERNO_MUCNID'] = df_xchange['ALTERNO'].apply(getMAlterno)


# In[21]:

# 2017-08
df_xchange['CHANGE_1_CNT'] = df_xchange['ALTDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xchange['CHANGE_2_CNT'] = df_xchange['ALTDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xchange['CHANGE_3_CNT'] = df_xchange['ALTDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xchange['CHANGE_5_CNT'] = df_xchange['ALTDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))



# In[22]:

# df_xchange.head()


# In[23]:

k = len(set(df_change['ALTERNO']))


# In[24]:

def totalcnt(x):
    return len(x)

df_xchange['ALTERNO_SUM'] = df_xchange['ALTERNO'].apply(totalcnt)

def cnt(x):
    return len(set(x))

df_xchange['ALTERNO_CNT'] = df_xchange['ALTERNO'].apply(cnt)

df_xchange['ALTERNO_CNT_RATE'] = df_xchange['ALTERNO_CNT'] / k

def getFirst(x):
    x.sort()
    return x[0]

df_xchange['FIRST_CHANGE_TIME'] = df_xchange['ALTDATE'].apply(getFirst)

def getEnd(x):
    x.sort()
    return x[-1]

df_xchange['END_CHANGE_TIME'] = df_xchange['ALTDATE'].apply(getEnd)

def equal(x):
    ok = 0
    for d in x:
        if d[:d.find(":")] != d[d.find(":")+1:]:
            ok = 1
    return ok
df_xchange['IS_CHANGE'] = (df_xchange['ALTBE'] +':'+ df_xchange['ALTAF']).apply(equal)


# In[25]:

df_xchange.to_csv('../data/public/2alter_1.csv',index=False,index_label=False)

# df_xchange = pd.read_csv('../data/public/2alter_1.csv')


# In[33]:

df_all = pd.merge(df_all,df_xchange[['EID', 'ALTERNO_SUM','ALTERNO_CNT', 'ALTERNO_CNT_RATE', 'FIRST_CHANGE_TIME',
                                    'END_CHANGE_TIME', 'IS_CHANGE','CHANGE_FIRST_ALTERNO','CHANGE_END_ALTERNO',
                                     'CHANGE_ALTERNO_MUCNID','CHANGE_1_CNT','CHANGE_2_CNT','CHANGE_3_CNT','CHANGE_5_CNT']],how='left',on=['EID'])


# In[34]:

# df_all.info()
# df_all.head()


# > 空值填充，根据HY的类别的平均值或者众数进行填充
#    + ['ALTERNO_SUM','ALTERNO_CNT', 'ALTERNO_CNT_RATE',]使用同一个大类别的均值进行填充
#    + ['FIRST_CHANGE_TIME','END_CHANGE_TIME', 'IS_CHANGE' ]使用同一个大类别的众数进行填充
# 
# 

# In[35]:

HYLIST = set(df_all['HY'])

# print(HYLIST)

meanlist = ['ALTERNO_SUM','ALTERNO_CNT', 'ALTERNO_CNT_RATE']
modelist = ['FIRST_CHANGE_TIME','END_CHANGE_TIME', 'IS_CHANGE','CHANGE_FIRST_ALTERNO','CHANGE_END_ALTERNO','CHANGE_ALTERNO_MUCNID' ]


# In[36]:


# int(df_all[df_all['HY']==75]['ALTERNO_SUM'].mean())

# df_all[df_all['HY']==7]['CHANGE_FIRST_ALTERNO'].value_counts().index[0]



# In[37]:

# for HY in HYLIST:
#     # print(df_train['HY'].value_counts())
#     for d in meanlist:
#         df_all.loc[df_all[df_all[d].isnull()][df_all['HY']==HY].index,d] = df_all[df_all['HY']==HY][d].mean()
    
#     for c in modelist:
#         if(len(df_all[df_all['HY']==HY][c].value_counts().index)==0): continue
#         df_all.loc[df_all[df_all[c].isnull()][df_all['HY']==HY].index,c] = df_all[df_all['HY']==HY][c].value_counts().index[0]


# In[38]:

# new 
df_all[['CHANGE_1_CNT','CHANGE_2_CNT','CHANGE_3_CNT','CHANGE_5_CNT']] = df_all[['CHANGE_1_CNT','CHANGE_2_CNT','CHANGE_3_CNT','CHANGE_5_CNT']].fillna(0)


# In[39]:

df_all[meanlist] = df_all[meanlist].fillna(0)


df_all[df_all['FIRST_CHANGE_TIME'].isnull()]['FIRST_CHANGE_TIME'] = df_all[df_all['FIRST_CHANGE_TIME'].isnull()]['RGYEAR'] 

df_all.loc[df_all['FIRST_CHANGE_TIME'].isnull(),'FIRST_CHANGE_TIME'] = df_all.loc[df_all['FIRST_CHANGE_TIME'].isnull(),'RGYEAR']
df_all.loc[df_all['END_CHANGE_TIME'].isnull(),'END_CHANGE_TIME'] = df_all.loc[df_all['END_CHANGE_TIME'].isnull(),'RGYEAR']


# In[ ]:




# In[40]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m
    
df_all['FIRST_CHANGE_TIME_DIFF'] = (df_all['FIRST_CHANGE_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['END_CHANGE_TIME_DIFF'] = (df_all['END_CHANGE_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)

df_all['END_FIRST_CHANGE_TIME_DIFF'] = (df_all['END_CHANGE_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)


# In[41]:

# df_all.info()

# 13. 平均每几个月变更一次事件，CHANGE_PRE_CNT = ALTERNO_SUM / END_CHANGE_TIME_DIFF
#     14. 每个企业变更事件的个数占所有企业变更事件平均值的比例， CHANGE_CNT_RATE = ALTERNO_SUM / AVG 
#     15. 企业对应大类HY的平均变更次数，CHANGE_HY_CNT_AVG
#     16. 企业对应大类HY的平均变更次数占所有变更时间的个数的平均值的比例，CHANGE_HY_CNT_ALL_AVG

# In[42]:

df_all['CHANGE_PRE_CNT'] = df_all['ALTERNO_SUM'] / df_all['END_CHANGE_TIME_DIFF']
df_all['CHANGE_CNT_RATE'] = df_all['ALTERNO_SUM'] / df_all['ALTERNO_SUM'].mean()


# In[43]:

tmp = pd.DataFrame()

tmp['CHANGE_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['ALTERNO_SUM'].mean()
tmp['CHANGE_HY_CNT_ALL_AVG'] = tmp['CHANGE_HY_CNT_AVG'] / df_all['ALTERNO_SUM'].mean()

tmp['CHANGE_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['ALTERNO_SUM'].max()
tmp['CHANGE_HY_CNT_ALL_MAX'] = tmp['CHANGE_HY_CNT_MAX'] / df_all['ALTERNO_SUM'].max()

tmp['HY'] = tmp.index



# In[44]:

tmp1 = pd.DataFrame()

tmp1['CHANGE_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['ALTERNO_SUM'].mean()
tmp1['CHANGE_ETYPE_CNT_ALL_AVG'] = tmp1['CHANGE_ETYPE_CNT_AVG'] / df_all['ALTERNO_SUM'].mean()

tmp1['CHANGE_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['ALTERNO_SUM'].max()
tmp1['CHANGE_ETYPE_CNT_ALL_MAX'] = tmp1['CHANGE_ETYPE_CNT_MAX'] / df_all['ALTERNO_SUM'].max()

tmp1['ETYPE'] = tmp1.index


# In[ ]:




# In[45]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])



# In[46]:

df_all['CHANGE_CNT_HY_RATE'] = df_all['ALTERNO_SUM'] / df_all['CHANGE_HY_CNT_AVG']
df_all['CHANGE_CNT_ETYPE_RATE'] = df_all['ALTERNO_SUM'] / df_all['CHANGE_ETYPE_CNT_AVG']

df_all['CHANGE_CNT_HY_RATE_MAX'] = df_all['ALTERNO_SUM'] / df_all['CHANGE_HY_CNT_MAX']
df_all['CHANGE_CNT_ETYPE_RATE_MAX'] = df_all['ALTERNO_SUM'] / df_all['CHANGE_ETYPE_CNT_MAX']



# In[47]:

# df_all.info()
# df_all.head()


# In[ ]:




# In[48]:

df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data12.csv',index=False,index_label=False)


# In[ ]:



