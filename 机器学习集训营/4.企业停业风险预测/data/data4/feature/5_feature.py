
# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 权利数据5right.csv提取特征
#     1. 企业拥有权利的个数，RIGHT_CNT
#     2. 企业拥有权利类型的个数，RIGHT_TYPE_CNT
#     3. 企业拥有权利类型的比例，RIGHT_TYPE_RATE
#     4. 第一个获得的权利的类型，RIGHT_FIRST_TYPECODE
#     5. 最后一个获得的权利的类型，RIGHT_END_TYPECODE
#     6. 获得最多的权利的类型，RIGHT_TYPECODE_MUCHID
# 
#     7. 第一个权利申请日期, RIGHT_FIRST_ASK_TIME
#     8. 第一个权利富于日期, RIGHT_FIRST_FB_TIME
#     9. 最后一个权利申请日期, RIGHT_END_ASK_TIME
#     10. 最后一个权利富于日期, RIGHT_END_FB_TIME
#     
#     11. 第一个权利申请日期和权利富于日期的差值，RIGHT_FIRST_ASK_FB_DIFF
#     12. 最后一个权利申请日期和权利富于日期的差值，RIGHT_END_ASK_FB_DIFF
#     13. 第一个和最后一个权利申请日期的差值，RIGHT_FIRST_END_ASK_DIFF
#     14. 第一个和最后一个权利富于日期的差值，RIGHT_FIRST_END_FB_DIFF
#     15. 第一个申请和最后一个权利富于日期的差值，RIGHT_FIRST_ASK_END_FB_DIFF
#     
#     16. 第一个权利的申请日期和公司注册时间的差值，RIGHT_FIRST_ASK_RGYEAR_DIFF
#     17. 第一个权利的富于日期和公司注册时间的差值，RIGHT_FIRST_FB_RGYEAR_DIFF
#     18. 最后一个权利的申请日期和公司注册时间的差值，RIGHT_END_ASK_RGYEAR_DIFF
#     19. 最后一个权利的富于日期和公司注册时间的差值，RIGHT_END_FB_RGYEAR_DIFF
#     
#     20. 第一个权利的申请日期和公司第一个变动时间的差值，RIGHT_FIRST_ASK_FIRST_CHANGE_DIFF
#     21. 第一个权利的富于日期和公司第一个变动时间的差值，RIGHT_FIRST_FB_FIRST_CHANGE_DIFF
#     22. 最后一个权利的申请日期和公司第一个变动时间的差值，RIGHT_END_ASK_FIRST_CHANGE_DIFF
#     23. 最后一个权利的富于日期和公司第一个变动时间的差值，RIGHT_END_FB_FIRST_CHANGE_DIFF
#     
#     24. 第一个权利的申请日期和公司最后一个变动时间的差值，RIGHT_FIRST_ASK_END_CHANGE_DIFF
#     25. 第一个权利的富于日期和公司最后一个变动时间的差值，RIGHT_FIRST_FB_END_CHANGE_DIFF
#     26. 最后一个权利的申请日期和公司最后一个变动时间的差值，RIGHT_END_ASK_END_CHANGE_DIFF
#     27. 最后一个权利的富于日期和公司最后一个变动时间的差值，RIGHT_END_FB_END_CHANGE_DIFF
#     
#     ---------------------------------------------------------------------------------------------------
#     28. 企业权利的个数占所有权利个数平均值的比例。RIGHT_CNT_ALL_RATE
#     
#     29. 企业对应的大类HY的平均权利的个数，RIGHT_HY_CNT_AVG
#     30. 企业对应大类HY的平均权利个数占所有权利平均个数的比例，RIGHT_HY_CNT_ALL_RATE
#     31. 企业权利的个数占其对应的大类HY的平均值的比例，RIGHT_CNT_HY_RATE
# 
#     -------------------------------------------------------------------------------------------------
#     
#     29. 企业对应的大类ETYPE的平均权利的个数，RIGHT_ETYPE_CNT_AVG
#     30. 企业对应大类ETYPE的平均权利个数占所有权利平均个数的比例，RIGHT_ETYPE_CNT_ALL_RATE
#     31. 企业权利的个数占其对应的大类ETYPE的平均值的比例，RIGHT_CNT_ETYPE_RATE
#     
#     添加对应的最大值的特征
#     
#     复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     RIGHT_K_OPEN_CNT, RIGHT_K_CLOSE_CNT
# 

# In[7]:

df_all = pd.read_csv("../data/alldata/df_data1234.csv")
df_right = pd.read_csv("../data/public/5right.csv")


# In[8]:

# df_all.info()
# df_all.head()


# In[9]:

# df_right.info()
# df_right.head()

df_right = df_right.sort_values(['ASKDATE','FBDATE'])


# In[10]:

# set(df_right['RIGHTTYPE'])
def settime(x):
    y = int(x[:x.find('-')])
    m = int(x[x.find('-')+1:])
    m+=2
    y = y+int(m/12)
    if m%12 == 0:
        m = 12
    else:
        m = m%12
    
    if(m<10):
        return str(y)+"-0"+str(m)
    
    return str(y)+"-"+str(m)


df_right.loc[df_right[df_right['RIGHTTYPE']==11][df_right['FBDATE'].isnull()].index,'FBDATE'] = df_right[df_right['RIGHTTYPE']==11][df_right['FBDATE'].isnull()]['ASKDATE']

df_right.loc[df_right[df_right['RIGHTTYPE']==40][df_right['FBDATE'].isnull()].index,'FBDATE'] = df_right[df_right['RIGHTTYPE']==40][df_right['FBDATE'].isnull()]['ASKDATE'].apply(settime)
df_right.loc[df_right[df_right['RIGHTTYPE']==50][df_right['FBDATE'].isnull()].index,'FBDATE'] = df_right[df_right['RIGHTTYPE']==50][df_right['FBDATE'].isnull()]['ASKDATE'].apply(settime)



# In[11]:

EIDS = set(df_right['EID'])

len(EIDS)


# In[12]:

columns = df_right.columns
df_xright = pd.DataFrame(columns=columns)

# print(columns)


# In[13]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_right[df_right['EID'] == EID]
    row = [EID,tmp['RIGHTTYPE'].values,tmp['TYPECODE'].values,tmp['ASKDATE'].values,tmp['FBDATE'].values]
    
    df_xright = df_xright.append(pd.Series(row,columns),ignore_index=True)
    


# In[14]:

# df_xright.info()
# df_xright.head()


# In[15]:

k = len(set(df_right['RIGHTTYPE']))

set(df_right['RIGHTTYPE'])


# In[16]:

def getTZCnt(x):
    return len(x)
    
df_xright['RIGHT_CNT'] = df_xright['RIGHTTYPE'].apply(getTZCnt)

def cnt(x):
    return len(set(x))

df_xright['RIGHT_TYPE_CNT'] = df_xright['RIGHTTYPE'].apply(getTZCnt)

df_xright['RIGHT_TYPE_RATE'] = df_xright['RIGHT_TYPE_CNT'] / k


# In[17]:

def getFAlterno(x):
    return x[0]

def getEAlterno(x):
    return x[-1]

def getMAlterno(x):
    x = list(x)
    x.sort()
    m = x[0]
    
    mk = 0
    mm = x[0]
    
    k = 1
    n = len(x)
    
    for i in range(1,n):
        if x[i] == x[i-1]:
            k+=1
        else:
            if k>mk:
                mk = k
                mm = x[i-1]
            
            k = 1
    return mm

df_xright['RIGHT_FIRST_TYPECODE'] = df_xright['RIGHTTYPE'].apply(getFAlterno)
df_xright['RIGHT_END_TYPECODE'] = df_xright['RIGHTTYPE'].apply(getEAlterno)
df_xright['RIGHT_TYPECODE_MUCHID'] = df_xright['RIGHTTYPE'].apply(getMAlterno)


# In[18]:

df_xright['RIGHT_FIRST_ASK_TIME'] = df_xright['ASKDATE'].apply(lambda x: x[0])
df_xright['RIGHT_FIRST_FB_TIME'] = df_xright['FBDATE'].apply(lambda x: x[0])

df_xright['RIGHT_END_ASK_TIME'] = df_xright['ASKDATE'].apply(lambda x: x[-1])
df_xright['RIGHT_END_FB_TIME'] = df_xright['FBDATE'].apply(lambda x: x[-1])



# In[ ]:




# In[19]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xright['RIGHT_1_OPEN_CNT'] = df_xright['ASKDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xright['RIGHT_2_OPEN_CNT'] = df_xright['ASKDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xright['RIGHT_3_OPEN_CNT'] = df_xright['ASKDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xright['RIGHT_5_OPEN_CNT'] = df_xright['ASKDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))

df_xright['RIGHT_1_CLOSE_CNT'] = df_xright['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xright['RIGHT_2_CLOSE_CNT'] = df_xright['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xright['RIGHT_3_CLOSE_CNT'] = df_xright['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xright['RIGHT_5_CLOSE_CNT'] = df_xright['FBDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))




# In[20]:

df_xright.to_csv('../data/public/5right_1.csv',index=False,index_label=False)
# df_xright.columns



# In[21]:

df_all = pd.merge(df_all,df_xright[['EID', 'RIGHT_CNT','RIGHT_TYPE_CNT', 'RIGHT_TYPE_RATE', 'RIGHT_FIRST_TYPECODE',
                                    'RIGHT_END_TYPECODE', 'RIGHT_TYPECODE_MUCHID', 'RIGHT_FIRST_ASK_TIME',
                                    'RIGHT_FIRST_FB_TIME', 'RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME',
                                   'RIGHT_1_OPEN_CNT','RIGHT_2_OPEN_CNT','RIGHT_3_OPEN_CNT','RIGHT_5_OPEN_CNT',
                                   'RIGHT_1_CLOSE_CNT','RIGHT_2_CLOSE_CNT','RIGHT_3_CLOSE_CNT','RIGHT_5_CLOSE_CNT']],how='left',on=['EID'])


# In[22]:

# 空值填充，根据HY的类别的平均值或者众数进行填充
# ['RIGHT_CNT','RIGHT_TYPE_CNT', 'RIGHT_TYPE_RATE']使用同一个大类别的均值进行填充
# ['RIGHT_FIRST_TYPECODE','RIGHT_END_TYPECODE', 'RIGHT_TYPECODE_MUCHID', 'RIGHT_FIRST_ASK_TIME',
#     'RIGHT_FIRST_FB_TIME', 'RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME']使用同一个大类别的众数进行填充

HYLIST = set(df_all['HY'])
# print(HYLIST)

meanlist = ['RIGHT_CNT','RIGHT_TYPE_CNT', 'RIGHT_TYPE_RATE']
modelist = ['RIGHT_FIRST_TYPECODE','RIGHT_END_TYPECODE', 'RIGHT_TYPECODE_MUCHID', 'RIGHT_FIRST_ASK_TIME',
            'RIGHT_FIRST_FB_TIME', 'RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME']


# In[23]:

for HY in HYLIST:
    # print(df_train['HY'].value_counts())
    for d in meanlist:
        df_all.loc[df_all[df_all[d].isnull()][df_all['HY']==HY].index,d] = df_all[df_all['HY']==HY][d].mean()
    
    for c in modelist:
        if(len(df_all[df_all['HY']==HY][c].value_counts().index)==0): continue
        df_all.loc[df_all[df_all[c].isnull()][df_all['HY']==HY].index,c] = df_all[df_all['HY']==HY][c].value_counts().index[0]


# In[24]:

df_all[['RIGHT_1_OPEN_CNT','RIGHT_2_OPEN_CNT','RIGHT_3_OPEN_CNT','RIGHT_5_OPEN_CNT']] = df_all[['RIGHT_1_OPEN_CNT','RIGHT_2_OPEN_CNT','RIGHT_3_OPEN_CNT','RIGHT_5_OPEN_CNT']].fillna(0)

df_all[['RIGHT_1_CLOSE_CNT','RIGHT_2_CLOSE_CNT','RIGHT_3_CLOSE_CNT','RIGHT_5_CLOSE_CNT']] = df_all[['RIGHT_1_CLOSE_CNT','RIGHT_2_CLOSE_CNT','RIGHT_3_CLOSE_CNT','RIGHT_5_CLOSE_CNT']].fillna(0)


# In[ ]:




# In[25]:

for d in meanlist:
    df_all.loc[df_all[d].isnull(),d] = 0

for c in modelist:
    df_all.loc[df_all[c].isnull(),c] = df_all[c].value_counts().index[0]


# In[ ]:




# In[26]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m



# In[27]:

df_all['RIGHT_FIRST_ASK_FB_DIFF'] = (df_all['RIGHT_FIRST_FB_TIME'] + ':' + df_all['RIGHT_FIRST_ASK_TIME']).apply(timeDiff)
df_all['RIGHT_END_ASK_FB_DIFF'] = (df_all['RIGHT_END_FB_TIME'] + ':' + df_all['RIGHT_END_ASK_TIME']).apply(timeDiff)
df_all['RIGHT_FIRST_END_ASK_DIFF'] = (df_all['RIGHT_END_ASK_TIME'] + ':' + df_all['RIGHT_FIRST_ASK_TIME']).apply(timeDiff)
df_all['RIGHT_FIRST_END_FB_DIFF'] = (df_all['RIGHT_END_FB_TIME'] + ':' + df_all['RIGHT_FIRST_FB_TIME']).apply(timeDiff)
df_all['RIGHT_FIRST_ASK_END_FB_DIFF'] = (df_all['RIGHT_END_FB_TIME'] + ':' + df_all['RIGHT_FIRST_ASK_TIME']).apply(timeDiff)


# In[28]:

df_all['RIGHT_FIRST_ASK_RGYEAR_DIFF'] = (df_all['RIGHT_FIRST_ASK_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['RIGHT_FIRST_FB_RGYEAR_DIFF'] = (df_all['RIGHT_FIRST_FB_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)

df_all['RIGHT_END_ASK_RGYEAR_DIFF'] = (df_all['RIGHT_END_ASK_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['RIGHT_END_FB_RGYEAR_DIFF'] = (df_all['RIGHT_END_FB_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)



# In[29]:

df_all['RIGHT_FIRST_ASK_FIRST_CHANGE_DIFF'] = (df_all['RIGHT_FIRST_ASK_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['RIGHT_FIRST_FB_FIRST_CHANGE_DIFF'] = (df_all['RIGHT_FIRST_FB_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)

df_all['RIGHT_END_ASK_FIRST_CHANGE_DIFF'] = (df_all['RIGHT_END_ASK_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['RIGHT_END_FB_FIRST_CHANGE_DIFF'] = (df_all['RIGHT_END_FB_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)



# In[30]:

df_all['RIGHT_FIRST_ASK_END_CHANGE_DIFF'] = (df_all['RIGHT_FIRST_ASK_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)
df_all['RIGHT_FIRST_FB_END_CHANGE_DIFF'] = (df_all['RIGHT_FIRST_FB_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)

df_all['RIGHT_END_ASK_END_CHANGE_DIFF'] = (df_all['RIGHT_END_ASK_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)
df_all['RIGHT_END_FB_END_CHANGE_DIFF'] = (df_all['RIGHT_END_FB_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)

    # ---------------------------------------------------------------------------------------------------
    # 28. 企业权利的个数占所有权利个数平均值的比例。RIGHT_CNT_ALL_RATE
    
    # 29. 企业对应的大类HY的平均权利的个数，RIGHT_HY_CNT_AVG
    # 30. 企业对应大类HY的平均权利个数占所有权利平均个数的比例，RIGHT_HY_CNT_ALL_RATE
    # 31. 企业权利的个数占其对应的大类HY的平均值的比例，RIGHT_CNT_HY_RATE

# In[31]:

df_all['RIGHT_CNT_ALL_RATE'] = df_all['RIGHT_CNT'] / df_all['RIGHT_CNT'].mean()
df_all['RIGHT_CNT_ALL_RATE_MAX'] = df_all['RIGHT_CNT'] / df_all['RIGHT_CNT'].max()



# In[ ]:




# In[ ]:




# In[32]:

tmp = pd.DataFrame()

tmp['RIGHT_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['RIGHT_CNT'].mean()
tmp['RIGHT_HY_CNT_ALL_RATE'] = tmp['RIGHT_HY_CNT_AVG'] / df_all['RIGHT_CNT'].mean()

tmp['RIGHT_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['RIGHT_CNT'].max()
tmp['RIGHT_HY_CNT_ALL_RATE'] = tmp['RIGHT_HY_CNT_MAX'] / df_all['RIGHT_CNT'].max()


tmp['HY'] = tmp.index


tmp1 = pd.DataFrame()

tmp1['RIGHT_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RIGHT_CNT'].mean()
tmp1['RIGHT_ETYPE_CNT_ALL_RATE'] = tmp1['RIGHT_ETYPE_CNT_AVG'] / df_all['RIGHT_CNT'].mean()

tmp1['RIGHT_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['RIGHT_CNT'].max()
tmp1['RIGHT_ETYPE_CNT_ALL_RATE'] = tmp1['RIGHT_ETYPE_CNT_MAX'] / df_all['RIGHT_CNT'].max()


tmp1['ETYPE'] = tmp1.index




# In[33]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])



# In[ ]:




# In[34]:

df_all['RIGHT_CNT_HY_RATE'] = df_all['RIGHT_CNT'] / df_all['RIGHT_HY_CNT_AVG']
df_all['RIGHT_CNT_HY_RATE_MAX'] = df_all['RIGHT_CNT'] / df_all['RIGHT_HY_CNT_MAX']

df_all['RIGHT_CNT_ETYPE_RATE'] = df_all['RIGHT_CNT'] / df_all['RIGHT_ETYPE_CNT_AVG']
df_all['RIGHT_CNT_ETYPE_RATE_MAX'] = df_all['RIGHT_CNT'] / df_all['RIGHT_ETYPE_CNT_MAX']



# In[35]:

# df_all.info()
# df_all.head()


# In[37]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data12345.csv',index=False,index_label=False)



# In[ ]:



