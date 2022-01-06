
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 投资数据4invest.csv(提取的特征)
# > 实验发现，所有被投资的企业在EID中都没有出现，所有随意每一个企业只有投资数据，没有被投资数据，而且不能构造新的训练数据
# 1. 企业投资企业的数量, TZ_CNT
# 2. 投资企业在省内的个数，TZ_INHOME_CNT
# 3. 投资企业在省内的个数的比例，TZ_INHOME_RATE
# 4. 投资企业在省外的个数，TZ_OUTHOME_CNT
# 5. 投资企业在省内的个数的比例，TZ_OUTHOME_RATE
# 6. 投资企业倒闭的个数，TZ_CLOSE_CNT
# 7. 投资企业倒闭的比例，TZ_CLOSE_RATE
# 8. 投资企业营业的个数，TZ_CLOSE_CNT
# 9. 投资企业营业的比例，TZ_CLOSE_RATE
# 10. 投资企业的平均持股比例, TZ_BTBL_AVG
# 11. 第一个投资企业的时间，TZ_QY_FIRST_TIME
# 12. 第一个投资企业的时间与本公司的注册时间的差值，TZ_QY_FIRST_RGYEAR_DIFF
# 13. 最后一个投资企业的时间，TZ_QY_END_TIME
# 14. 最后一个投资企业的时间与本公司的注册时间的差值，TZ_QY_END_RGYEAR_DIFF 
#     
#              
#                  
# x 15. 第一个投资企业的时间与本公司的第一个分公司注册时间的差值，TZ_QY_FIRST_BRANCH_FIRST_DIFF
# x 16. 第一个投资企业的时间与本公司的最后一个分公司注册时间的差值，TZ_QY_FIRST_BRANCH_END_DIFF
# 
# 17. 第一个投资企业倒闭的时间，TZ_FIRST_CLOSE_TIME
# 18. 第一个投资企业倒闭的时间距离第一企业的投资时间，TZ_FIRST_CLOSE_TIME
# 19. 第一个投资企业倒闭的时间距离本企业的注册时间，TZ_FIRST_CLOSE_RGYEAR_TIME
# 20. 该企业是否有投资企业，IS_TZ_ACTIVE
# 
# ---------------------------------------------------------------------------------------------
# + 20. 平均每几个月投资一个企业，TZ_PRE_MONTH_CNT = TZ_CNT / TZ_QY_END_RGYEAR_DIFF
# + 21. 平均每几个月投资的企业倒闭，TZ_CLOSR_PRE_MONTH_CNT = TZ_CLOSE_CNR / TZ_QY_END_RGYEAR_DIFF
# 
# 
# + 23. 企业对应的HY大类的平均投资数量，TZ_HY_CNT_AVG
# + 24. 企业的投资数量占其对应投资大类HY平均值的比例, TZ_CNT_HY_AVG_RATE
# + 25. 企业的投资数量占其所有投资平均值的比例, TZ_CNT_ALL_AVG_RATE
# + 26. 企业的对应的行业大类HY的平均投资投资数量占其所有投资平均值的比例, TZ_HY_CNT_ALL_AVG_RATE
# 
# + 23. 企业对应的HY大类的平均投资close数量，TZ_HY_CNT_AVG
# + 25. 企业的投资数量占其所有投资close平均值的比例, TZ_CNT_ALL_AVG_RATE
# + 26. 企业的对应的行业大类HY的平均投close资投资数量占其所有投资平均值的比例, TZ_HY_CNT_ALL_AVG_RATE
# 
# + 23. 企业对应的HY大类的平均投资open数量，TZ_HY_CNT_AVG
# + 25. 企业的投资数量占其所有投资open平均值的比例, TZ_CNT_ALL_AVG_RATE
# + 26. 企业的对应的行业大类HY的平均open投资投资数量占其所有投资平均值的比例, TZ_HY_CNT_ALL_AVG_RATE
# 
# 
# 投资企业的平均持股比例, TZ_BTBL_AVG
# 
# + 23. 企业对应的HY大类的持股比例的均值，TZ_HY_BTBL_AVG
# + 24. 企业的投资持股比例占其对应投资大类HY平均值的比例, TZ_BTBL_HY_AVG_RATE
# + 25. 企业的投资持股比例占其所有投资平均值的比例, TZ_BTBL_ALL_AVG_RATE
# + 26. 企业的对应的行业大类HY的投资比例均值占其所有投资平均值的比例, TZ_HY_BTBL_ALL_AVG_RATE
# 
# 复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
# 时间是开始是2017-08之前的k年
# 1. 之前k年的变更时间的个数。
# TZ_K_OPEN_CNT, TZ_K_CLOSE_CNT
# 
# 
-----------------------------------------------------------------------------------
+ 23. 企业对应的ETYPE大类的平均投资数量，TZ_ETYPE_CNT_AVG
+ 24. 企业的投资数量占其对应投资大类ETYPE平均值的比例, TZ_CNT_ETYPE_AVG_RATE
+ 25. 企业的投资数量占其所有投资平均值的比例, TZ_CNT_ALL_AVG_RATE
+ 26. 企业的对应的行业大类ETYPE的平均投资投资数量占其所有投资平均值的比例, TZ_ETYPE_CNT_ALL_AVG_RATE

+ 23. 企业对应的ETYPE大类的平均投资close数量，TZ_ETYPE_CNT_AVG
+ 25. 企业的投资数量占其所有投资close平均值的比例, TZ_CNT_ALL_AVG_RATE
+ 26. 企业的对应的行业大类ETYPE的平均投close资投资数量占其所有投资平均值的比例, TZ_ETYPE_CNT_ALL_AVG_RATE

+ 23. 企业对应的ETYPE大类的持股比例的均值，TZ_ETYPE_BTBL_AVG
+ 24. 企业的投资持股比例占其对应投资大类ETYPE平均值的比例, TZ_BTBL_ETYPE_AVG_RATE
+ 25. 企业的投资持股比例占其所有投资平均值的比例, TZ_BTBL_ALL_AVG_RATE
+ 26. 企业的对应的行业大类ETYPE的投资比例均值占其所有投资平均值的比例, TZ_ETYPE_BTBL_ALL_AVG_RATE

# In[6]:

df_all = pd.read_csv("../data/alldata/df_data123.csv")
df_invest = pd.read_csv("../data/public/4invest.csv")


# In[7]:

# 实验发现，所有被投资的企业在EID中出现的个数，
# tmp = df_invest['BTEID']
# df_all[df_all['EID'].isin(tmp)]


# In[8]:

# df_invest.info()
# df_invest.head()


# In[9]:

# 正在运行的企业，用2018.0填充
df_invest['BTENDYEAR'] = df_invest['BTENDYEAR'].fillna(2018.0)

df_invest = df_invest.sort_values(['BTYEAR','BTENDYEAR'])

columns = df_invest.columns
df_xinvest = pd.DataFrame(columns=columns)


# In[10]:

EIDS = set(df_invest['EID'])

len(EIDS)


# In[11]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_invest[df_invest['EID'] == EID]
    row = [EID,tmp['BTEID'].values,tmp['IFHOME'].values,tmp['BTBL'].values,tmp['BTYEAR'].values,tmp['BTENDYEAR'].values]
    
    df_xinvest = df_xinvest.append(pd.Series(row,columns),ignore_index=True)
    


# In[12]:

# df_xinvest.to_csv('../data/public/4invert_1.csv',index=False,index_label=False)
# df_xinvest = pd.read_csv('../data/public/3invert_1.csv')


# In[13]:

# df_xinvest.info()
# df_xinvest.head()


# In[14]:

def getTZCnt(x):
    return len(x)
    
df_xinvest['TZ_CNT'] = df_xinvest['BTEID'].apply(getTZCnt)


# In[15]:

def InHome(x):
    return list(x).count(1)

def OutHome(x):
    return list(x).count(0)

df_xinvest['TZ_INHOME_CNT'] = df_xinvest['IFHOME'].apply(InHome)
df_xinvest['TZ_OUTHOME_CNT'] = df_xinvest['IFHOME'].apply(OutHome)

df_xinvest['TZ_INHOME_RATE'] = df_xinvest['TZ_INHOME_CNT'] / df_xinvest['TZ_CNT']
df_xinvest['TZ_OUTHOME_RATE'] = df_xinvest['TZ_OUTHOME_CNT'] / df_xinvest['TZ_CNT']


# In[16]:

def EndCnt(x):
    k = 0
    for d in x:
        if d != 2018.0:
            k+=1
    return k

def OpenCnt(x):
    k = 0
    for d in x:
        if d == 2018.0:
            k+=1
    return k

df_xinvest['TZ_CLOSE_CNT'] = df_xinvest['BTENDYEAR'].apply(EndCnt)
df_xinvest['TZ_OPEN_CNT'] = df_xinvest['BTENDYEAR'].apply(OpenCnt)

df_xinvest['TZ_CLOSE_RATE'] = df_xinvest['TZ_CLOSE_CNT'] / df_xinvest['TZ_CNT']
df_xinvest['TZ_OPEN_RATE'] = df_xinvest['TZ_OPEN_CNT'] / df_xinvest['TZ_CNT']


# In[17]:

df_xinvest['TZ_BTBL_AVG'] = df_xinvest['BTBL'].apply(np.mean)
df_xinvest['IS_TZ_ACTIVE'] = 1


# In[18]:

def getFirstTime(x):
    return str(int(x[0]))+'-01'

def getEndTime(x):
    return str(int(x[-1]))+'-01'

df_xinvest['TZ_QY_FIRST_TIME'] = df_xinvest['BTYEAR'].apply(getFirstTime)
df_xinvest['TZ_QY_END_TIME'] = df_xinvest['BTYEAR'].apply(getEndTime)
df_xinvest['TZ_FIRST_CLOSE_TIME'] = df_xinvest['BTENDYEAR'].apply(getFirstTime)



# In[20]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xinvest['TZ_1_OPEN_CNT'] = df_xinvest['BTYEAR'].map(lambda x: np.sum(np.array(x) >= 2016))
df_xinvest['TZ_2_OPEN_CNT'] = df_xinvest['BTYEAR'].map(lambda x: np.sum(np.array(x) >= 2015))
df_xinvest['TZ_3_OPEN_CNT'] = df_xinvest['BTYEAR'].map(lambda x: np.sum(np.array(x) >= 2014))
df_xinvest['TZ_5_OPEN_CNT'] = df_xinvest['BTYEAR'].map(lambda x: np.sum(np.array(x) >= 2012))

df_xinvest['TZ_1_CLOSE_CNT'] = df_xinvest['BTENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2016))
df_xinvest['TZ_2_CLOSE_CNT'] = df_xinvest['BTENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2015))
df_xinvest['TZ_3_CLOSE_CNT'] = df_xinvest['BTENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2014))
df_xinvest['TZ_5_CLOSE_CNT'] = df_xinvest['BTENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2012))




# In[ ]:




# In[21]:

df_xinvest.to_csv('../data/public/4invert_1.csv',index=False,index_label=False)
# df_xinvest.columns


# In[22]:

df_all = pd.merge(df_all,df_xinvest[['EID', 'TZ_CNT','TZ_INHOME_CNT', 'TZ_OUTHOME_CNT', 'TZ_INHOME_RATE', 'TZ_OUTHOME_RATE',
                                     'TZ_CLOSE_CNT', 'TZ_OPEN_CNT', 'TZ_CLOSE_RATE', 'TZ_OPEN_RATE','TZ_BTBL_AVG', 
                                     'TZ_QY_FIRST_TIME', 'TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME','IS_TZ_ACTIVE',
                                    'TZ_1_OPEN_CNT','TZ_2_OPEN_CNT','TZ_3_OPEN_CNT','TZ_5_OPEN_CNT',
                                    'TZ_1_CLOSE_CNT','TZ_2_CLOSE_CNT','TZ_3_CLOSE_CNT','TZ_5_CLOSE_CNT']],how='left',on=['EID'])


# In[23]:

# df_all.head()


# In[24]:

# 空值填充，根据HY的类别的平均值或者众数进行填充
# ['TZ_CNT','TZ_INHOME_CNT', 'TZ_OUTHOME_CNT','TZ_INHOME_RATE','TZ_OUTHOME_RATE', 'TZ_CLOSE_CNT', 'TZ_OPEN_CNT','TZ_CLOSE_RATE', 'TZ_OPEN_RATE','TZ_BTBL_AVG']使用同一个大类别的均值进行填充
# ['TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME']使用同一个大类别的众数进行填充
HYLIST = set(df_all['HY'])

# print(HYLIST)

meanlist = ['TZ_CNT','TZ_INHOME_CNT', 'TZ_OUTHOME_CNT','TZ_INHOME_RATE','TZ_OUTHOME_RATE', 'TZ_CLOSE_CNT', 'TZ_OPEN_CNT','TZ_CLOSE_RATE', 'TZ_OPEN_RATE','TZ_BTBL_AVG']
modelist = ['TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME']


# In[25]:

# for HY in HYLIST:
#     # print(df_train['HY'].value_counts())
#     for d in meanlist:
#         df_all.loc[df_all[df_all[d].isnull()][df_all['HY']==HY].index,d] = df_all[df_all['HY']==HY][d].mean()
    
#     for c in modelist:
#         if(len(df_all[df_all['HY']==HY][c].value_counts().index)==0): continue
#         df_all.loc[df_all[df_all[c].isnull()][df_all['HY']==HY].index,c] = df_all[df_all['HY']==HY][c].value_counts().index[0]


# In[26]:

# meanlist中的空值，表示企业没有投资数据，使用0进行填充
df_all[meanlist] = df_all[meanlist].fillna(0)

# df_all['TZ_QY_FIRST_TIME'] = df_all['TZ_QY_FIRST_TIME'].fillna(df_all['TZ_QY_FIRST_TIME'].value_counts().index[0],)
# df_all['TZ_QY_END_TIME'] = df_all['TZ_QY_END_TIME'].fillna(df_all['TZ_QY_END_TIME'].value_counts().index[0],)
# df_all['TZ_FIRST_CLOSE_TIME'] = df_all['TZ_FIRST_CLOSE_TIME'].fillna(df_all['TZ_FIRST_CLOSE_TIME'].value_counts().index[0],)


# In[27]:

df_all[['TZ_1_OPEN_CNT','TZ_2_OPEN_CNT','TZ_3_OPEN_CNT','TZ_5_OPEN_CNT']] = df_all[['TZ_1_OPEN_CNT','TZ_2_OPEN_CNT','TZ_3_OPEN_CNT','TZ_5_OPEN_CNT']].fillna(0)

df_all[['TZ_1_CLOSE_CNT','TZ_2_CLOSE_CNT','TZ_3_CLOSE_CNT','TZ_5_CLOSE_CNT']] = df_all[['TZ_1_CLOSE_CNT','TZ_2_CLOSE_CNT','TZ_3_CLOSE_CNT','TZ_5_CLOSE_CNT']].fillna(0)


# In[28]:

# 所有关于年的空值使用注册时间填充
df_all.loc[df_all['TZ_QY_FIRST_TIME'].isnull(),'TZ_QY_FIRST_TIME'] = df_all.loc[df_all['TZ_QY_FIRST_TIME'].isnull(),'RGYEAR']
df_all.loc[df_all['TZ_QY_END_TIME'].isnull(),'TZ_QY_END_TIME'] = df_all.loc[df_all['TZ_QY_END_TIME'].isnull(),'RGYEAR']
df_all.loc[df_all['TZ_FIRST_CLOSE_TIME'].isnull(),'TZ_FIRST_CLOSE_TIME'] = df_all.loc[df_all['TZ_FIRST_CLOSE_TIME'].isnull(),'RGYEAR']


# In[29]:

df_all['IS_TZ_ACTIVE'] = df_all['IS_TZ_ACTIVE'].fillna(0)


# In[30]:

# df_all.info()
# df_all.head()


# In[ ]:




# In[ ]:




# In[31]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m


# In[32]:

df_all['TZ_QY_FIRST_RGYEAR_DIFF'] = (df_all['TZ_QY_FIRST_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['TZ_QY_END_RGYEAR_DIFF'] = (df_all['TZ_QY_END_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)


df_all['TZ_QY_FIRST_BRANCH_FIRST_DIFF'] = (df_all['TZ_QY_FIRST_TIME'] + ':' + df_all['BRANCH_FIRST_YEAR']).apply(timeDiff)
df_all['TZ_QY_FIRST_BRANCH_END_DIFF'] = (df_all['TZ_QY_FIRST_TIME'] + ':' + df_all['BRANCH_END_YEAR']).apply(timeDiff)

df_all['TZ_FIRST_CLOSE_TIME_DIFF'] = (df_all['TZ_FIRST_CLOSE_TIME'] + ':' + df_all['TZ_QY_FIRST_TIME']).apply(timeDiff)
df_all['TZ_FIRST_CLOSE_RGYEAR_TIME_DIFF'] = (df_all['TZ_FIRST_CLOSE_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)


# In[33]:

df_all['TZ_PRE_MONTH_CNT '] = df_all['TZ_CNT'] / df_all['TZ_QY_END_RGYEAR_DIFF']
df_all['TZ_CLOSR_PRE_MONTH_CNT  '] = df_all['TZ_CLOSE_CNT'] / df_all['TZ_QY_END_RGYEAR_DIFF']


# In[34]:


df_all['TZ_CNT_ALL_AVG_RATE'] = df_all['TZ_CNT'] / df_all['TZ_CNT'].mean()
df_all['TZ_BTBL_ALL_AVG_RATE'] = df_all['TZ_BTBL_AVG'] / df_all['TZ_BTBL_AVG'].mean()


# In[35]:

tmp = pd.DataFrame()

tmp['TZ_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_CNT'].mean()
tmp['TZ_HY_CNT_ALL_AVG_RATE'] = tmp['TZ_HY_CNT_AVG'] / df_all['TZ_CNT'].mean()

tmp['TZ_CLOSE_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_CLOSE_CNT'].mean()
tmp['TZ_HY_CLOSE_CNT_ALL_AVG_RATE'] = tmp['TZ_CLOSE_HY_CNT_AVG'] / df_all['TZ_CLOSE_CNT'].mean()

tmp['TZ_OPEN_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_OPEN_CNT'].mean()
tmp['TZ_HY_OPEN_CNT_ALL_AVG_RATE'] = tmp['TZ_OPEN_HY_CNT_AVG'] / df_all['TZ_OPEN_CNT'].mean()

tmp['TZ_HY_BTBL_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_BTBL_AVG'].mean()
tmp['TZ_HY_BTBL_ALL_AVG_RATE'] = tmp['TZ_HY_BTBL_AVG'] / df_all['TZ_BTBL_AVG'].mean()


tmp['TZ_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_CNT'].max()
tmp['TZ_HY_CNT_ALL_MAX_RATE'] = tmp['TZ_HY_CNT_MAX'] / df_all['TZ_CNT'].max()

tmp['TZ_CLOSE_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_CLOSE_CNT'].max()
tmp['TZ_HY_CLOSE_CNT_ALL_MAX_RATE'] = tmp['TZ_CLOSE_HY_CNT_MAX'] / df_all['TZ_CLOSE_CNT'].max()

tmp['TZ_OPEN_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_OPEN_CNT'].max()
tmp['TZ_HY_OPEN_CNT_ALL_MAX_RATE'] = tmp['TZ_OPEN_HY_CNT_MAX'] / df_all['TZ_OPEN_CNT'].max()

tmp['TZ_HY_BTBL_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['TZ_BTBL_AVG'].max()
tmp['TZ_HY_BTBL_ALL_MAX_RATE'] = tmp['TZ_HY_BTBL_MAX'] / df_all['TZ_BTBL_AVG'].max()



tmp['HY'] = tmp.index



# In[ ]:




# In[36]:

tmp1 = pd.DataFrame()

tmp1['TZ_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_CNT'].mean()
tmp1['TZ_ETYPE_CNT_ALL_AVG_RATE'] = tmp1['TZ_ETYPE_CNT_AVG'] / df_all['TZ_CNT'].mean()

tmp1['TZ_CLOSE_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_CLOSE_CNT'].mean()
tmp1['TZ_ETYPE_CLOSE_CNT_ALL_AVG_RATE'] = tmp1['TZ_CLOSE_ETYPE_CNT_AVG'] / df_all['TZ_CLOSE_CNT'].mean()

tmp1['TZ_OPEN_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_OPEN_CNT'].mean()
tmp1['TZ_ETYPE_OPEN_CNT_ALL_AVG_RATE'] = tmp1['TZ_OPEN_ETYPE_CNT_AVG'] / df_all['TZ_OPEN_CNT'].mean()

tmp1['TZ_ETYPE_BTBL_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_BTBL_AVG'].mean()
tmp1['TZ_ETYPE_BTBL_ALL_AVG_RATE'] = tmp1['TZ_ETYPE_BTBL_AVG'] / df_all['TZ_BTBL_AVG'].mean()


tmp1['TZ_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_CNT'].max()
tmp1['TZ_ETYPE_CNT_ALL_MAX_RATE'] = tmp1['TZ_ETYPE_CNT_MAX'] / df_all['TZ_CNT'].max()

tmp1['TZ_CLOSE_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_CLOSE_CNT'].max()
tmp1['TZ_ETYPE_CLOSE_CNT_ALL_MAX_RATE'] = tmp1['TZ_CLOSE_ETYPE_CNT_MAX'] / df_all['TZ_CLOSE_CNT'].max()

tmp1['TZ_OPEN_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_OPEN_CNT'].max()
tmp1['TZ_ETYPE_OPEN_CNT_ALL_MAX_RATE'] = tmp1['TZ_OPEN_ETYPE_CNT_MAX'] / df_all['TZ_OPEN_CNT'].max()

tmp1['TZ_ETYPE_BTBL_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['TZ_BTBL_AVG'].max()
tmp1['TZ_ETYPE_BTBL_ALL_MAX_RATE'] = tmp1['TZ_ETYPE_BTBL_MAX'] / df_all['TZ_BTBL_AVG'].max()


tmp1['ETYPE'] = tmp1.index


# In[37]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])


# In[38]:

df_all['TZ_CNT_HY_AVG_RATE'] = df_all['TZ_CNT'] / df_all['TZ_HY_CNT_AVG']
df_all['TZ_BTBL_HY_AVG_RATE'] = df_all['TZ_BTBL_AVG'] / df_all['TZ_HY_BTBL_AVG']

df_all['TZ_CNT_ETYPE_AVG_RATE'] = df_all['TZ_CNT'] / df_all['TZ_ETYPE_CNT_AVG']
df_all['TZ_BTBL_ETYPE_AVG_RATE'] = df_all['TZ_BTBL_AVG'] / df_all['TZ_ETYPE_BTBL_AVG']


df_all['TZ_CNT_HY_MAX_RATE'] = df_all['TZ_CNT'] / df_all['TZ_HY_CNT_MAX']
df_all['TZ_BTBL_HY_MAX_RATE'] = df_all['TZ_BTBL_AVG'] / df_all['TZ_HY_BTBL_MAX']

df_all['TZ_CNT_ETYPE_MAX_RATE'] = df_all['TZ_CNT'] / df_all['TZ_ETYPE_CNT_MAX']
df_all['TZ_BTBL_ETYPE_MAX_RATE'] = df_all['TZ_BTBL_AVG'] / df_all['TZ_ETYPE_BTBL_MAX']




# In[ ]:




# In[39]:

# df_all.info()
# df_all.head()


# In[40]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data1234.csv',index=False,index_label=False)




# In[ ]:



