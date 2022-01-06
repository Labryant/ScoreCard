
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 招聘数据9recruit.csv 提取特征
#     1. 企业发布招聘的个数，RECRUIT_CNT
#     2. 企业发布招聘平台的个数，RECRUIT_WZCODE_CNT
#     3. 企业发布的招聘数量的均值，RECRUIT_RECRNUM_AVG
#     
#     4. 企业发布的招聘数量的和，RECRUIT_RECRNUM_SUM
#     
#     5. 企业发布的招聘数量的最多的个数，RECRUIT_RECRNUM_MAX
#     6. 企业发布的招聘数量的最少的个数，RECRUIT_RECRNUM_MIN
#     7. 企业发布第一个招聘的日期，RECRUIT_FIRST_TIME
#     8. 企业发布最后一个招聘的日期，RECRUIT_END_TIME
#     
#     9. 企业发布最后一个招聘的日期和第一个招聘日期的差值，RECRUIT_END_FIRST_TIME_DIFF
#     10. 企业发布最后一个招聘的日期和注册日期的差值，RECRUIT_END_RGYEAR_TIME_DIFF
#     
#     ---------------------------------------------------------------------------------------
#     11. 企业平均一次的招聘几个， RECRUIT_PRE_MONTH_CNT_AVG = RECRUIT_RECRNUM_SUM / RECRUIT_CNT
#     12. 企业招聘的个数占所有招聘个数个平均值的比例，RECRUIT_CNT_ALL_RATE
#     
#     13. 企业对应的大类HY的平均招聘数据的个数，RECRUIT_HY_CNT_AVG
#     14. 企业对应大类HY的平均招聘个数占所有招聘平均个数的比例，RECRUIT_HY_CNT_ALL_RATE
#     15. 企业招聘的个数占其对应的大类HY的招聘平均值的比例，RECRUIT_CNT_HY_RATE
# 
# 
#     16. 企业招聘的数量占所有招聘数量平均值的比例，RECRUIT_RECRNUM_AVG_ALL_RATE
#     
#     17. 企业对应的大类HY的平均招聘数据的平均，RECRUIT_HY_RECRNUM_AVG_AVG
#     18. 企业对应大类HY的平均招聘数量占所有招聘平均数量的比例，RECRUIT_HY_RECRNUM_AVG_ALL_RATE
#     19. 企业招聘的数量占其对应的大类HY的招聘数量平均值的比例，RECRUIT_RECRNUM_AVG_HY_RATE
# 
# 
#     20. 企业招聘的数量占所有招聘数量平均值的比例，RECRUIT_RECRNUM_SUM_ALL_RATE
#     
#     21. 企业对应的大类HY的平均招聘数据的平均，RECRUIT_HY_RECRNUM_SUM_AVG
#     22. 企业对应大类HY的平均招聘数量占所有招聘平均数量的比例，RECRUIT_HY_RECRNUM_SUM_ALL_RATE
#     23. 企业招聘的数量占其对应的大类HY的招聘数量平均值的比例，RECRUIT_RECRNUM_SUM_HY_RATE
# 
#     ---------------------------------------------------------------------------------------
# 
#     13. 企业对应的大类ETYPE的平均招聘数据的个数，RECRUIT_ETYPE_CNT_AVG
#     14. 企业对应大类ETYPE的平均招聘个数占所有招聘平均个数的比例，RECRUIT_ETYPE_CNT_ALL_RATE
#     15. 企业招聘的个数占其对应的大类ETYPE的招聘平均值的比例，RECRUIT_CNT_ETYPE_RATE
# 
#     
#     17. 企业对应的大类ETYPE的平均招聘数据的平均，RECRUIT_ETYPE_RECRNUM_AVG_AVG
#     18. 企业对应大类ETYPE的平均招聘数量占所有招聘平均数量的比例，RECRUIT_ETYPE_RECRNUM_AVG_ALL_RATE
#     19. 企业招聘的数量占其对应的大类ETYPE的招聘数量平均值的比例，RECRUIT_RECRNUM_AVG_ETYPE_RATE
# 
#     
#     21. 企业对应的大类ETYPE的平均招聘数据的平均，RECRUIT_ETYPE_RECRNUM_SUM_AVG
#     22. 企业对应大类ETYPE的平均招聘数量占所有招聘平均数量的比例，RECRUIT_ETYPE_RECRNUM_SUM_ALL_RATE
#     23. 企业招聘的数量占其对应的大类ETYPE的招聘数量平均值的比例，RECRUIT_RECRNUM_SUM_ETYPE_RATE
# 
#     
#     添加对应的max数据
#     
#     new feature POSCOD, 首先将 PNUM --> RECRNUM, 删除POSCO的特征 
#     
#     复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     RECRUIT_K_OPEN_CNT,
#     

# In[4]:




# In[10]:

df_all = pd.read_csv("../data/alldata/df_data123456789.csv")
df_recruit = pd.read_csv("../data/public/9recruit.csv")


# In[14]:

# df_all.info()
# df_all.head()


# In[3]:

# df_recruit.info()
# df_recruit.head()



# In[5]:

df_recruit.rename(columns={'PNUM':'RECRNUM'},inplace=True)
df_recruit.drop('POSCODE',axis=1,inplace=True)

df_recruit['RECRNUM'] = df_recruit['RECRNUM'].map(lambda x: str(x).replace('若干','30').replace('人',''))

df_recruit = df_recruit.sort_values(['RECDATE'])


# In[6]:

# df_recruit.info()
# df_recruit.head()

# df_recruit['WZCODE'].value_counts()


# In[7]:

df_recruit.loc[df_recruit['RECRNUM']=='nan','RECRNUM'] = 0


# In[8]:

df_recruit['RECRNUM'] = df_recruit['RECRNUM'].fillna(0)
df_recruit['RECRNUM'] = df_recruit['RECRNUM'].astype(np.int)


# In[9]:

EIDS = set(df_recruit['EID'])

# print(len(EIDS))

columns = df_recruit.columns
df_xrecruit = pd.DataFrame(columns=columns)

# print(columns)


# In[10]:

# df_recruit['WZCODE'].value_counts()


# In[11]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_recruit[df_recruit['EID'] == EID]
    row = [EID,tmp['WZCODE'].values,tmp['RECDATE'].values,tmp['RECRNUM'].values]
    
    df_xrecruit = df_xrecruit.append(pd.Series(row,columns),ignore_index=True)
    


# In[12]:

# df_xrecruit.info()
# df_xrecruit.head()


# In[13]:

# df_xrecruit.rename(columns={'RECDATE':'RECDATE1','RECRNUM':'RECRNUM1'},inplace=True)

# df_xrecruit.rename(columns={'RECDATE1':'RECRNUM','RECRNUM1':'RECDATE'},inplace=True)


# In[14]:

def fun(x):
    x = list(x)
    if 'nan' in x:
        x.remove('nan')
    
    x = np.array(x).astype(int)
    
    return x

df_xrecruit['RECRNUM'] = df_xrecruit['RECRNUM'].apply(fun)


# In[15]:

df_xrecruit['RECRUIT_CNT'] = df_xrecruit['WZCODE'].apply(lambda x: len(x))
df_xrecruit['RECRUIT_WZCODE_CNT'] = df_xrecruit['WZCODE'].apply(lambda x: len(set(x)))

df_xrecruit['RECRUIT_RECRNUM_AVG'] = df_xrecruit['RECRNUM'].apply(lambda x: np.mean(x))
df_xrecruit['RECRUIT_RECRNUM_SUM'] = df_xrecruit['RECRNUM'].apply(lambda x: sum(x))
df_xrecruit['RECRUIT_RECRNUM_MAX'] = df_xrecruit['RECRNUM'].apply(lambda x: max(x))
df_xrecruit['RECRUIT_RECRNUM_MIN'] = df_xrecruit['RECRNUM'].apply(lambda x: min(x))

def getFirstTIme(x):
    x = list(x)
    x.sort()
    return x[0]

def getEndTIme(x):
    x = list(x)
    x.sort()
    return x[-1]

df_xrecruit['RECRUIT_FIRST_TIME'] = df_xrecruit['RECDATE'].apply(getFirstTIme)
df_xrecruit['RECRUIT_END_TIME'] = df_xrecruit['RECDATE'].apply(getEndTIme)


# In[17]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xrecruit['RECRUIT_1_OPEN_CNT'] = df_xrecruit['RECDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xrecruit['RECRUIT_2_OPEN_CNT'] = df_xrecruit['RECDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xrecruit['RECRUIT_3_OPEN_CNT'] = df_xrecruit['RECDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xrecruit['RECRUIT_5_OPEN_CNT'] = df_xrecruit['RECDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))



# In[ ]:




# In[18]:

# df_xrecruit.info()
# df_xrecruit.head()


# In[19]:

df_xrecruit.to_csv('../data/public/9recruit_1.csv',index=False,index_label=False)
df_xrecruit.columns



# In[20]:

df_all = pd.merge(df_all,df_xrecruit[['EID','RECRUIT_CNT','RECRUIT_WZCODE_CNT', 'RECRUIT_RECRNUM_AVG', 'RECRUIT_RECRNUM_SUM',
                                      'RECRUIT_RECRNUM_MAX', 'RECRUIT_RECRNUM_MIN', 'RECRUIT_FIRST_TIME',
                                      'RECRUIT_END_TIME','RECRUIT_1_OPEN_CNT','RECRUIT_2_OPEN_CNT',
                                      'RECRUIT_3_OPEN_CNT','RECRUIT_5_OPEN_CNT']],how='left',on=['EID'])


# In[21]:

# 空值填充，根据HY的类别的平均值或者众数进行填充
# ['RIGHT_CNT','RIGHT_TYPE_CNT', 'RIGHT_TYPE_RATE']使用同一个大类别的均值进行填充
# ['RIGHT_FIRST_TYPECODE','RIGHT_END_TYPECODE', 'RIGHT_TYPECODE_MUCHID', 'RIGHT_FIRST_ASK_TIME',
#     'RIGHT_FIRST_FB_TIME', 'RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME']使用同一个大类别的众数进行填充

HYLIST = set(df_all['HY'])
# print(HYLIST)

meanlist = ['RECRUIT_CNT','RECRUIT_WZCODE_CNT', 'RECRUIT_RECRNUM_AVG', 'RECRUIT_RECRNUM_SUM','RECRUIT_RECRNUM_MAX','RECRUIT_RECRNUM_MIN']
modelist = ['RECRUIT_FIRST_TIME','RECRUIT_END_TIME']


# In[22]:

for HY in HYLIST:
    # print(df_train['HY'].value_counts())
    for d in meanlist:
        df_all.loc[df_all[df_all[d].isnull()][df_all['HY']==HY].index,d] = df_all[df_all['HY']==HY][d].mean()
    
    for c in modelist:
        if(len(df_all[df_all['HY']==HY][c].value_counts().index)==0): continue
        df_all.loc[df_all[df_all[c].isnull()][df_all['HY']==HY].index,c] = df_all[df_all['HY']==HY][c].value_counts().index[0]


# In[23]:

for d in meanlist:
    df_all.loc[df_all[d].isnull(),d] = 0

for c in modelist:
    df_all.loc[df_all[c].isnull(),c] = df_all[c].value_counts().index[0]


# In[24]:

df_all[['RECRUIT_1_OPEN_CNT','RECRUIT_2_OPEN_CNT','RECRUIT_3_OPEN_CNT','RECRUIT_5_OPEN_CNT']] = df_all[['RECRUIT_1_OPEN_CNT','RECRUIT_2_OPEN_CNT','RECRUIT_3_OPEN_CNT','RECRUIT_5_OPEN_CNT']].fillna(0)


# In[25]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m


df_all['RECRUIT_END_FIRST_TIME_DIFF'] = (df_all['RECRUIT_END_TIME'] + ':' + df_all['RECRUIT_FIRST_TIME']).apply(timeDiff)
df_all['RECRUIT_END_RGYEAR_TIME_DIFF'] = (df_all['RECRUIT_END_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)

# -------------------------------------------------------------------------------------
#     11. 企业平均一次的招聘几个， RECRUIT_PRE_MONTH_CNT_AVG = RECRUIT_RECRNUM_SUM / RECRUIT_CNT
#     12. 企业招聘的个数占所有招聘个数个平均值的比例，RECRUIT_CNT_ALL_RATE
    
#     13. 企业对应的大类HY的平均招聘数据的个数，RECRUIT_HY_CNT_AVG
#     14. 企业对应大类HY的平均招聘个数占所有招聘平均个数的比例，RECRUIT_HY_CNT_ALL_RATE
#     15. 企业招聘的个数占其对应的大类HY的招聘平均值的比例，RECRUIT_CNT_HY_RATE

#     RECRUIT_RECRNUM_AVG
#     16. 企业招聘的数量占所有招聘数量平均值的比例，RECRUIT_RECRNUM_AVG_ALL_RATE
    
#     17. 企业对应的大类HY的平均招聘数据的平均，RECRUIT_HY_RECRNUM_AVG_AVG
#     18. 企业对应大类HY的平均招聘数量占所有招聘平均数量的比例，RECRUIT_HY_RECRNUM_AVG_ALL_RATE
#     19. 企业招聘的数量占其对应的大类HY的招聘数量平均值的比例，RECRUIT_RECRNUM_AVG_HY_RATE

    # RECRUIT_RECRNUM_SUM
    # 20. 企业招聘的数量占所有招聘数量平均值的比例，RECRUIT_RECRNUM_SUM_ALL_RATE
    
    # 21. 企业对应的大类HY的平均招聘数据的平均，RECRUIT_HY_RECRNUM_SUM_AVG
    # 22. 企业对应大类HY的平均招聘数量占所有招聘平均数量的比例，RECRUIT_HY_RECRNUM_SUM_ALL_RATE
    # 23. 企业招聘的数量占其对应的大类HY的招聘数量平均值的比例，RECRUIT_RECRNUM_SUM_HY_RATE

# In[26]:

df_all['RECRUIT_PRE_MONTH_CNT_AVG'] = df_all['RECRUIT_RECRNUM_SUM'] / df_all['RECRUIT_CNT']

df_all['RECRUIT_CNT_ALL_RATE'] = df_all['RECRUIT_CNT'] / df_all['RECRUIT_CNT'].mean()
df_all['RECRUIT_RECRNUM_AVG_ALL_RATE'] = df_all['RECRUIT_RECRNUM_AVG'] / df_all['RECRUIT_RECRNUM_AVG'].mean()
df_all['RECRUIT_RECRNUM_SUM_ALL_RATE'] = df_all['RECRUIT_RECRNUM_SUM'] / df_all['RECRUIT_RECRNUM_SUM'].mean()

df_all['RECRUIT_CNT_ALL_RATE_MAX'] = df_all['RECRUIT_CNT'] / df_all['RECRUIT_CNT'].max()
df_all['RECRUIT_RECRNUM_AVG_ALL_RATE_MAX'] = df_all['RECRUIT_RECRNUM_AVG'] / df_all['RECRUIT_RECRNUM_AVG'].max()
df_all['RECRUIT_RECRNUM_SUM_ALL_RATE_MAX'] = df_all['RECRUIT_RECRNUM_SUM'] / df_all['RECRUIT_RECRNUM_SUM'].max()



# In[27]:


tmp = pd.DataFrame()

tmp['RECRUIT_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['RECRUIT_CNT'].mean()
tmp['RECRUIT_HY_CNT_ALL_RATE'] = tmp['RECRUIT_HY_CNT_AVG'] / df_all['RECRUIT_CNT'].mean()

tmp['RECRUIT_HY_RECRNUM_AVG_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['RECRUIT_RECRNUM_AVG'].mean()
tmp['RECRUIT_HY_RECRNUM_AVG_ALL_RATE'] = tmp['RECRUIT_HY_RECRNUM_AVG_AVG'] / df_all['RECRUIT_RECRNUM_AVG'].mean()

tmp['RECRUIT_HY_RECRNUM_SUM_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['RECRUIT_RECRNUM_SUM'].mean()
tmp['RECRUIT_HY_RECRNUM_SUM_ALL_RATE'] = tmp['RECRUIT_HY_RECRNUM_SUM_AVG'] / df_all['RECRUIT_RECRNUM_SUM'].mean()



tmp['RECRUIT_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['RECRUIT_CNT'].mean()
tmp['RECRUIT_HY_CNT_ALL_RATE_MAX'] = tmp['RECRUIT_HY_CNT_MAX'] / df_all['RECRUIT_CNT'].mean()

tmp['RECRUIT_HY_RECRNUM_AVG_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['RECRUIT_RECRNUM_AVG'].mean()
tmp['RECRUIT_HY_RECRNUM_AVG_ALL_RATE_MAX'] = tmp['RECRUIT_HY_RECRNUM_AVG_MAX'] / df_all['RECRUIT_RECRNUM_AVG'].mean()

tmp['RECRUIT_HY_RECRNUM_SUM_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['RECRUIT_RECRNUM_SUM'].mean()
tmp['RECRUIT_HY_RECRNUM_SUM_ALL_RATE_MAX'] = tmp['RECRUIT_HY_RECRNUM_SUM_MAX'] / df_all['RECRUIT_RECRNUM_SUM'].mean()


tmp['HY'] = tmp.index



# In[28]:


tmp1 = pd.DataFrame()

tmp1['RECRUIT_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RECRUIT_CNT'].mean()
tmp1['RECRUIT_ETYPE_CNT_ALL_RATE'] = tmp1['RECRUIT_ETYPE_CNT_AVG'] / df_all['RECRUIT_CNT'].mean()

tmp1['RECRUIT_ETYPE_RECRNUM_AVG_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RECRUIT_RECRNUM_AVG'].mean()
tmp1['RECRUIT_ETYPE_RECRNUM_AVG_ALL_RATE'] = tmp1['RECRUIT_ETYPE_RECRNUM_AVG_AVG'] / df_all['RECRUIT_RECRNUM_AVG'].mean()

tmp1['RECRUIT_ETYPE_RECRNUM_SUM_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RECRUIT_RECRNUM_SUM'].mean()
tmp1['RECRUIT_ETYPE_RECRNUM_SUM_ALL_RATE'] = tmp1['RECRUIT_ETYPE_RECRNUM_SUM_AVG'] / df_all['RECRUIT_RECRNUM_SUM'].mean()



tmp1['RECRUIT_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RECRUIT_CNT'].mean()
tmp1['RECRUIT_ETYPE_CNT_ALL_RATE_MAX'] = tmp1['RECRUIT_ETYPE_CNT_MAX'] / df_all['RECRUIT_CNT'].mean()

tmp1['RECRUIT_ETYPE_RECRNUM_AVG_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RECRUIT_RECRNUM_AVG'].mean()
tmp1['RECRUIT_ETYPE_RECRNUM_AVG_ALL_RATE_MAX'] = tmp1['RECRUIT_ETYPE_RECRNUM_AVG_MAX'] / df_all['RECRUIT_RECRNUM_AVG'].mean()

tmp1['RECRUIT_ETYPE_RECRNUM_SUM_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RECRUIT_RECRNUM_SUM'].mean()
tmp1['RECRUIT_ETYPE_RECRNUM_SUM_ALL_RATE_MAX'] = tmp1['RECRUIT_ETYPE_RECRNUM_SUM_MAX'] / df_all['RECRUIT_RECRNUM_SUM'].mean()


tmp1['ETYPE'] = tmp1.index



# In[29]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])


# In[30]:

df_all['RECRUIT_CNT_HY_RATE'] = df_all['RECRUIT_CNT'] / df_all['RECRUIT_HY_CNT_AVG']
df_all['RECRUIT_RECRNUM_AVG_HY_RATE'] = df_all['RECRUIT_RECRNUM_AVG'] / df_all['RECRUIT_HY_RECRNUM_AVG_AVG']
df_all['RECRUIT_RECRNUM_SUM_HY_RATE'] = df_all['RECRUIT_RECRNUM_SUM'] / df_all['RECRUIT_HY_RECRNUM_SUM_AVG']

df_all['RECRUIT_CNT_HY_RATE_MAX'] = df_all['RECRUIT_CNT'] / df_all['RECRUIT_HY_CNT_MAX']
df_all['RECRUIT_RECRNUM_AVG_HY_RATE_MAX'] = df_all['RECRUIT_RECRNUM_AVG'] / df_all['RECRUIT_HY_RECRNUM_AVG_MAX']
df_all['RECRUIT_RECRNUM_SUM_HY_RATE_MAX'] = df_all['RECRUIT_RECRNUM_SUM'] / df_all['RECRUIT_HY_RECRNUM_SUM_MAX']


# In[31]:

df_all['RECRUIT_CNT_ETYPE_RATE'] = df_all['RECRUIT_CNT'] / df_all['RECRUIT_ETYPE_CNT_AVG']
df_all['RECRUIT_RECRNUM_AVG_ETYPE_RATE'] = df_all['RECRUIT_RECRNUM_AVG'] / df_all['RECRUIT_ETYPE_RECRNUM_AVG_AVG']
df_all['RECRUIT_RECRNUM_SUM_ETYPE_RATE'] = df_all['RECRUIT_RECRNUM_SUM'] / df_all['RECRUIT_ETYPE_RECRNUM_SUM_AVG']

df_all['RECRUIT_CNT_ETYPE_RATE_MAX'] = df_all['RECRUIT_CNT'] / df_all['RECRUIT_ETYPE_CNT_MAX']
df_all['RECRUIT_RECRNUM_AVG_ETYPE_RATE_MAX'] = df_all['RECRUIT_RECRNUM_AVG'] / df_all['RECRUIT_ETYPE_RECRNUM_AVG_MAX']
df_all['RECRUIT_RECRNUM_SUM_ETYPE_RATE_MAX'] = df_all['RECRUIT_RECRNUM_SUM'] / df_all['RECRUIT_ETYPE_RECRNUM_SUM_MAX']



# In[15]:

# df_all.info()
# df_all.head()


# In[16]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data123456789.csv',index=False,index_label=False)



# In[ ]:




# In[ ]:



