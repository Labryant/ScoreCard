
# coding: utf-8

# In[10]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# + 第一个修改特征
# # 项目数据6project.csv提取的特征
#     1. 项目的数量个数， PROJECT_CNT,
#     2. 省内项目的个数， PROJECT_INHOME_CNT
#     3. 省外项目的个数， PROJECT_OUTHOME_CNT
#     4. 省内项目的比例， PROJECT_INHOME_RATE
#     5. 省外项目的比例， PROJECT_OUTHOME_RATE
#     
#     6. 项目的类型数目，PROJECT_TYPECODE_CNT
# 
#     7. 第一个项目的项目ID，PROJECT_FIRST_TYPECODE
#     8. 最后一个项目的项目ID，PROJECT_END_TYPECODE
#     
#     9. 第一个项目中标的时间，PROJECT_FIRST_TIME
#     10. 最后一个项目中标的时间，PROJECT_END_TIME
#     
#     11. 最后一个项目中标时间和第一个中标时间的差值，PROJECT_END_FIRST_DIFF
#     12. 第一个项目中标时间距离企业注册时间的差值，PROJECT_FIRST_RGYEAR_DIFF
#     13. 最后一个项目中标时间距离企业注册时间的差值，PROJECT_END_RGYEAR_DIFF
# 
# 
#     14. 第一个项目中标时间距离企业第一个变更时间的差值，PROJECT_FIRST_CHANGE_FIRST_DIFF
#     15. 最后一个项目中标时间距离企第一个变更时间的差值，PROJECT_END_CHANGE_FIRST_DIFF
#     16. 第一个项目中标时间距离企业最后一个变更时间的差值，PROJECT_FIRST_CHANGE_END_DIFF
#     17. 最后一个项目中标时间距离企最后一个变更时间的差值，PROJECT_END_CHANGE_END_DIFF
#     
#     ---------------------------------------------------------------------------------
#     18. 平均没几个月中标一个项目其余用0表示，PROJECT_PRE_MONTH_CNT = PROJECT_END_RGYEAR_DIFF / PROJECT_CNT
# 
#     19. 企业权利的个数占所有权利个数平均值的比例。PROJECT_CNT_ALL_RATE
#     
#     20. 企业对应的大类HY的平均权利的个数，PROJECT_HY_CNT_AVG
#     21. 企业对应大类HY的平均权利个数占所有权利平均个数的比例，PROJECT_HY_CNT_ALL_RATE
#     22. 企业权利的个数占其对应的大类HY的平均值的比例，PROJECT_CNT_HY_RATE
# 
#     ---------------------------------------------------------------------------------
#   
#     19. 企业权利的个数占所有权利个数最大值的比例。PROJECT_CNT_ALL_RATE_MAX
#     
#     20. 企业对应的大类ETYPE的平均权利的个数，PROJECT_ETYPE_CNT_AVG
#     21. 企业对应大类ETYPE的平均权利个数占所有权利平均个数的比例，PROJECT_ETYPE_CNT_ALL_RATE
#     22. 企业权利的个数占其对应的大类ETYPE的平均值的比例，PROJECT_CNT_ETYPE_RATE
#     
#     添加对应的MAX特征数据
#      复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     RIGHT_K_OPEN_CNT, RIGHT_K_CLOSE_CNT
#     
#     

# In[11]:

df_all = pd.read_csv("../data/alldata/df_data12345.csv")
df_project = pd.read_csv("../data/public/6project.csv")


# In[12]:

# df_all.info()
# df_all.head()


# In[13]:

# df_project.info()
# df_project.head()



# In[ ]:




# In[15]:

EIDS = set(df_project['EID'])

# print(len(EIDS))

columns = df_project.columns
df_xproject = pd.DataFrame(columns=columns)

# print(columns)


# In[16]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_project[df_project['EID'] == EID]
    row = [EID,tmp['TYPECODE'].values,tmp['DJDATE'].values,tmp['IFHOME'].values]
    
    df_xproject = df_xproject.append(pd.Series(row,columns),ignore_index=True)
    


# In[17]:

# df_xproject.info()
# df_xproject.head()


# In[18]:

df_xproject['PROJECT_CNT'] = df_xproject['TYPECODE'].apply(lambda x: len(x))
df_xproject['PROJECT_INHOME_CNT'] = df_xproject['IFHOME'].apply(lambda x: list(x).count(1))
df_xproject['PROJECT_OUTHOME_CNT'] = df_xproject['IFHOME'].apply(lambda x: list(x).count(0))


# In[19]:

df_xproject['PROJECT_INHOME_RATE'] = df_xproject['PROJECT_INHOME_CNT'] / df_xproject['PROJECT_CNT']
df_xproject['PROJECT_OUTHOME_RATE'] = df_xproject['PROJECT_OUTHOME_CNT'] / df_xproject['PROJECT_CNT']



# In[20]:

df_xproject['PROJECT_TYPECODE_CNT'] = df_xproject['TYPECODE'].apply(lambda x: len(set(x)) )

df_xproject['PROJECT_FIRST_TYPECODE'] = df_xproject['TYPECODE'].apply(lambda x: x[0])
df_xproject['PROJECT_END_TYPECODE'] = df_xproject['TYPECODE'].apply(lambda x: x[-1])

df_xproject['PROJECT_FIRST_TIME'] = df_xproject['DJDATE'].apply(lambda x: x[0])
df_xproject['PROJECT_END_TIME'] = df_xproject['DJDATE'].apply(lambda x: x[-1])


# In[21]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xproject['PROJECT_1_OPEN_CNT'] = df_xproject['DJDATE'].map(lambda x: np.sum(np.array(x) >= '2016-08'))
df_xproject['PROJECT_2_OPEN_CNT'] = df_xproject['DJDATE'].map(lambda x: np.sum(np.array(x) >= '2015-08'))
df_xproject['PROJECT_3_OPEN_CNT'] = df_xproject['DJDATE'].map(lambda x: np.sum(np.array(x) >= '2014-08'))
df_xproject['PROJECT_5_OPEN_CNT'] = df_xproject['DJDATE'].map(lambda x: np.sum(np.array(x) >= '2012-08'))



# In[22]:

df_xproject.to_csv('../data/public/6project_1.csv',index=False,index_label=False)
# df_xproject.columns



# In[23]:

df_all = pd.merge(df_all,df_xproject[['EID', 'PROJECT_CNT','PROJECT_INHOME_CNT', 'PROJECT_OUTHOME_CNT', 'PROJECT_INHOME_RATE',
                                    'PROJECT_OUTHOME_RATE', 'PROJECT_TYPECODE_CNT', 'PROJECT_FIRST_TYPECODE', 'PROJECT_END_TYPECODE',
                                    'PROJECT_FIRST_TIME','PROJECT_END_TIME','PROJECT_1_OPEN_CNT','PROJECT_2_OPEN_CNT',
                                      'PROJECT_3_OPEN_CNT','PROJECT_5_OPEN_CNT']],how='left',on=['EID'])


# In[24]:

# df_all.info()
# df_all.head()


# In[ ]:




# In[25]:

# 空值填充，根据HY的类别的平均值或者众数进行填充
# ['RIGHT_CNT','RIGHT_TYPE_CNT', 'RIGHT_TYPE_RATE']使用同一个大类别的均值进行填充
# ['RIGHT_FIRST_TYPECODE','RIGHT_END_TYPECODE', 'RIGHT_TYPECODE_MUCHID', 'RIGHT_FIRST_ASK_TIME',
#     'RIGHT_FIRST_FB_TIME', 'RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME']使用同一个大类别的众数进行填充

HYLIST = set(df_all['HY'])
# print(HYLIST)

meanlist = ['PROJECT_FIRST_TYPECODE', 'PROJECT_END_TYPECODE','PROJECT_CNT','PROJECT_INHOME_CNT', 'PROJECT_OUTHOME_CNT','PROJECT_OUTHOME_CNT','PROJECT_OUTHOME_RATE','PROJECT_TYPECODE_CNT']
modelist = ['PROJECT_FIRST_TIME', 'PROJECT_END_TIME']


# In[26]:

df_all[['PROJECT_1_OPEN_CNT','PROJECT_2_OPEN_CNT','PROJECT_3_OPEN_CNT','PROJECT_5_OPEN_CNT']] = df_all[['PROJECT_1_OPEN_CNT','PROJECT_2_OPEN_CNT','PROJECT_3_OPEN_CNT','PROJECT_5_OPEN_CNT']].fillna(0)

for HY in HYLIST:
    # print(df_train['HY'].value_counts())
    for d in meanlist:
        df_all.loc[df_all[df_all[d].isnull()][df_all['HY']==HY].index,d] = df_all[df_all['HY']==HY][d].mean()
    
    for c in modelist:
        if(len(df_all[df_all['HY']==HY][c].value_counts().index)==0): continue
        df_all.loc[df_all[df_all[c].isnull()][df_all['HY']==HY].index,c] = df_all[df_all['HY']==HY][c].value_counts().index[0]
for d in meanlist:
    df_all.loc[df_all[d].isnull(),d] = df_all[d].mean()

for c in modelist:
    df_all.loc[df_all[c].isnull(),c] = df_all[c].value_counts().index[0]

df_all = df_all.fillna(0)
# In[27]:

# 所有的关于时间的空值用2018-01填充，假设这些企业被执行的数据发生在下一年，其他用0填充便是没有被执行

for d in meanlist:
    df_all.loc[df_all[d].isnull(),d] = 0

for c in modelist:
    df_all.loc[df_all[c].isnull(),c] = '2018-01'


# In[28]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m


# In[29]:

df_all['PROJECT_END_FIRST_DIFF'] = (df_all['PROJECT_END_TIME'] + ':' + df_all['PROJECT_FIRST_TIME']).apply(timeDiff)

df_all['PROJECT_FIRST_RGYEAR_DIFF'] = (df_all['PROJECT_FIRST_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['PROJECT_END_RGYEAR_DIFF'] = (df_all['PROJECT_END_TIME'] + ':' + df_all['RGYEAR']).apply(timeDiff)

df_all['PROJECT_FIRST_CHANGE_FIRST_DIFF'] = (df_all['PROJECT_FIRST_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)
df_all['PROJECT_END_CHANGE_FIRST_DIFF'] = (df_all['PROJECT_END_TIME'] + ':' + df_all['FIRST_CHANGE_TIME']).apply(timeDiff)

df_all['PROJECT_FIRST_CHANGE_END_DIFF'] = (df_all['PROJECT_FIRST_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)
df_all['PROJECT_END_CHANGE_END_DIFF'] = (df_all['PROJECT_END_TIME'] + ':' + df_all['END_CHANGE_TIME']).apply(timeDiff)

    # ---------------------------------------------------------------------------------
    # 18. 平均没几个月中标一个项目其余用0表示，PROJECT_PRE_MONTH_CNT = PROJECT_END_RGYEAR_DIFF / PROJECT_CNT

    # 19. 企业权利的个数占所有权利个数平均值的比例。PROJECT_CNT_ALL_RATE
    
    # 20. 企业对应的大类HY的平均权利的个数，PROJECT_HY_CNT_AVG
    # 21. 企业对应大类HY的平均权利个数占所有权利平均个数的比例，PROJECT_HY_CNT_ALL_RATE
    # 22. 企业权利的个数占其对应的大类HY的平均值的比例，PROJECT_CNT_HY_RATE

# In[30]:

df_all['PROJECT_PRE_MONTH_CNT'] = df_all['PROJECT_END_RGYEAR_DIFF'] / df_all['PROJECT_CNT']
df_all['PROJECT_CNT_ALL_RATE'] = df_all['PROJECT_CNT'] / df_all['PROJECT_CNT'].mean()

df_all['PROJECT_CNT_ALL_RATE_MAX'] = df_all['PROJECT_CNT'] / df_all['PROJECT_CNT'].max()


# In[31]:

tmp = pd.DataFrame()

tmp['PROJECT_HY_CNT_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['PROJECT_CNT'].mean()
tmp['PROJECT_HY_CNT_ALL_RATE'] = tmp['PROJECT_HY_CNT_AVG'] / df_all['PROJECT_CNT'].mean()

tmp['PROJECT_HY_CNT_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['PROJECT_CNT'].max()
tmp['PROJECT_HY_CNT_ALL_RATE_MAX'] = tmp['PROJECT_HY_CNT_MAX'] / df_all['PROJECT_CNT'].max()

tmp['HY'] = tmp.index


# In[32]:

tmp1 = pd.DataFrame()

tmp1['PROJECT_ETYPE_CNT_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['PROJECT_CNT'].mean()
tmp1['PROJECT_ETYPE_CNT_ALL_RATE'] = tmp1['PROJECT_ETYPE_CNT_AVG'] / df_all['PROJECT_CNT'].mean()

tmp1['PROJECT_ETYPE_CNT_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['PROJECT_CNT'].max()
tmp1['PROJECT_ETYPE_CNT_ALL_RATE_MAX'] = tmp1['PROJECT_ETYPE_CNT_MAX'] / df_all['PROJECT_CNT'].max()

tmp1['ETYPE'] = tmp1.index




# In[33]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])



# In[34]:

df_all['PROJECT_CNT_HY_RATE'] = df_all['PROJECT_CNT'] / df_all['PROJECT_HY_CNT_AVG']
df_all['PROJECT_CNT_HY_RATE_MAX'] = df_all['PROJECT_CNT'] / df_all['PROJECT_HY_CNT_MAX']

df_all['PROJECT_CNT_ETYPE_RATE'] = df_all['PROJECT_CNT'] / df_all['PROJECT_ETYPE_CNT_AVG']
df_all['PROJECT_CNT_ETYPE_RATE_MAX'] = df_all['PROJECT_CNT'] / df_all['PROJECT_ETYPE_CNT_MAX']


# In[ ]:




# In[35]:


# df_all.info()
# df_all.head()


# In[36]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)
df_all.to_csv('../data/alldata/df_data123456.csv',index=False,index_label=False)



# In[21]:




# In[ ]:



