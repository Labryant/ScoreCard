
# coding: utf-8

# In[15]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 分支结构的特征3branch.csv 特征
#     1. 分支机构的数量， BRANCH_CNT
#     2. 分支结构在本省的数量, BRANCH_CNT_INHOME
#     3. 分支结构在外省的数量, BRANCH_CNT_OUTHOME
#     4. 第一个分支机构成立时间与总部成立时间的差（单位月）,BRANCH_FIRST_YEAR_RGYEAR_DIFF
#     4.1 第一个分支机构成立时间与总部成立时间, BRANCH_FIRST_YEAR
#     5. 最后一个分支机构成立时间与总部成立时间的差（单位月）, BRANCH_END_YEAR_RGYEAR_DIFF
#     5.1 最后一个分支机构成立时间与总部成立时间, BRANCH_END_YEAR
#     6. 平均每几个月成立一个分支结构，（没有分支结构用-1表示）, PER_BRANCH_MONTH= 最后一个分支机构成立时间与总部成立时间的差/分支机构的数量     
#     7. 分支结构关停的个数，BRANCH_CLOSE_CNT
#     8. 分支结构关停的比例，(关停个数/总数), BRANCH_CLOSE_RATE
#     9. 分支结构正常运行的个数，BRANCH_OPEN_CNT
#     10. 分支结构正常运行的比例，(运行个数/总数), BRANCH_OPEN_RATE
#     11. 第一个分支机构关停时间距离总部注册时间的差值，BRANCH_FIRST_CLOSE_YEAR_RGYEAR_DIFF
#     11.1  第一个分支机构关停时间 BRANCH_FIRST_CLOSE_YEAR
#     12. 第一个分支机构关停时间距离第一个分支机构成立时间的差值，BRANCH_FIRST_CLOSE_YEAR_FIRST_DIFF
#     13. 第一个分支机构关停时间距离最后一个分支机构成立时间的差值，BRANCH_FIRST_CLOSE_YEAR_END_DIFF
# 
# -----------------------------------------------------------------------------------------------
#     14. 分支结构的数量占所有分支机构平均值的比例, BRANCH_CNT_ALL_RATE = BRANCH_CNT / 所有分支数据的平均值
#     
#     15. 企业所在大类HY的对应的总分支机构的平均值，BRANCH_HY_AVG
#     16. 企业所在大类HY的对应的分支机构的平均值占所有机构的平均值的比例，BRANCH_HY_ALL_AVG_RATE
#     
#     00. 企业分支机构个数占对于大类HY的平均值的比例，BRANCH_HY_AVG_RATE
#     
#     
#     17. 分支结构的关停数量占所有关停分支机构平均值的比例, BRANCH_CLOSE_CNT_ALL_RATE = BRANCH_CLOSE_CNT / 所有关停分支数据的平均值
#     18. 企业所在大类HY的对应的关停分支机构的平均值，BRANCH_CLOSE_HY_AVG
#     19. 企业所在大类HY的对应的关停分支机构的平均值占所有关停机构的平均值的比例，BRANCH_HY_CLOSE_ALL_AVG_RATE
#     00. 企业分支机构个数占对于大类HY的平均值的比例，BRANCH_HY_CLOSE_AVG_RATE
#     
#     
#     
#     17. 分支结构的营业数量占所有营业分支机构平均值的比例, BRANCH_OPEN_CNT_ALL_RATE = BRANCH_OPEN_CNT / 所有营业分支数据的平均值
#     18. 企业所在大类HY的对应的营业分支机构的平均值，BRANCH_OPEN_HY_AVG
#     19. 企业所在大类HY的对应的营业分支机构的平均值占所有营业机构的平均值的比例，BRANCH_HY_OPEN_ALL_AVG_RATE
#     00. 企业分支机构个数占对于大类HY的平均值的比例，BRANCH_HY_OPEN_AVG_RATE
#     
#     
#     
#     20. 平均几个月关闭一个分支结构，PER_BRANCH_CLOSE_MONTH = BRANCH_CLOSE_CNT / BRANCH_END_YEAR_RGYEAR_DIFF 
#     
#     -----------------------------------------------------------------------------------------
#     15. 企业所在大类ETYPE的对应的总分支机构的平均值，BRANCH_ETYPE_AVG
#     16. 企业所在大类ETYPE的对应的分支机构的平均值占所有机构的平均值的比例，BRANCH_ETYPE_ALL_AVG_RATE
#     00. 企业分支机构个数占对于大类HY的平均值的比例，BRANCH_ETYPE_AVG_RATE
#     
#     对应的最大值的比例的个数，
# 
#     复赛新特征，滑动窗口系列的特征，时间段是1年，2年，3年，5年，计算在最近k[1,2,3,5]年之后的数据，主要是个数和次数。
#     时间是开始是2017-08之前的k年
#     1. 之前k年的变更时间的个数。
#     BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
# 
# 
# 

# In[16]:

df_all = pd.read_csv("../data/alldata/df_data12.csv")
df_branch = pd.read_csv("../data/public/3branch.csv")


# In[17]:

# df_branch['B_ENDYEAR'].sort_values()


# In[18]:

# df_all.info()


# In[19]:

# df_branch.info()
# df_branch.head()


# In[20]:

# 正在运行的企业，用2018.0填充
df_branch['B_ENDYEAR'] = df_branch['B_ENDYEAR'].fillna(2018.0)

columns = df_branch.columns
df_xbranch = pd.DataFrame(columns=columns)


# In[21]:

# df_all.head()


# In[22]:

EIDS = set(df_branch['EID'])

len(EIDS)


# In[23]:

k = 0
for EID in EIDS:
    if k%3000 == 0:
        print('第%d次处理--------->',k)
    k+=1
    tmp = df_branch[df_branch['EID'] == EID]
    row = [EID,tmp['TYPECODE'].values,tmp['IFHOME'].values,tmp['B_REYEAR'].values,tmp['B_ENDYEAR'].values]
    
    df_xbranch = df_xbranch.append(pd.Series(row,columns),ignore_index=True)
    


# In[24]:

# df_xbranch.head()


# In[ ]:




# In[25]:

def TotalCnt(x):
    return len(x)

df_xbranch['BRANCH_CNT'] = df_xbranch['TYPECODE'].apply(TotalCnt)


# In[26]:

def InHome(x):
    return list(x).count(1)

def OutHome(x):
    return list(x).count(0)

df_xbranch['BRANCH_CNT_INHOME'] = df_xbranch['IFHOME'].apply(InHome)
df_xbranch['BRANCH_CNT_OUTHOME'] = df_xbranch['IFHOME'].apply(OutHome)


# In[27]:

def getFirstTime(x):
    return str(x[0])+'-01'

def getEndTime(x):
    return str(x[-1])+'-01'

df_xbranch['BRANCH_FIRST_YEAR'] = df_xbranch['B_REYEAR'].apply(getFirstTime)
df_xbranch['BRANCH_END_YEAR'] = df_xbranch['B_REYEAR'].apply(getEndTime)



# In[28]:

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

df_xbranch['BRANCH_CLOSE_CNT'] = df_xbranch['B_ENDYEAR'].apply(EndCnt)
df_xbranch['BRANCH_OPEN_CNT'] = df_xbranch['B_ENDYEAR'].apply(OpenCnt)

df_xbranch['BRANCH_CLOSE_RATE'] = df_xbranch['BRANCH_CLOSE_CNT'] / df_xbranch['BRANCH_CNT']
df_xbranch['BRANCH_OPEN_RATE'] = df_xbranch['BRANCH_OPEN_CNT'] / df_xbranch['BRANCH_CNT']


# In[29]:

# 倒闭的情况下才能计算
def FirstClostYear(x):
    list(x).sort()
    return str(int(x[0]))+'-01'

df_xbranch['BRANCH_FIRST_CLOSE_YEAR'] = df_xbranch['B_ENDYEAR'].apply(FirstClostYear)


# In[30]:

# 2017-08  BRANCH_K_OPEN_CNT, BRANCH_K_CLOSE_CNT
df_xbranch['BRANCH_1_OPEN_CNT'] = df_xbranch['B_REYEAR'].map(lambda x: np.sum(np.array(x) >= 2016))
df_xbranch['BRANCH_2_OPEN_CNT'] = df_xbranch['B_REYEAR'].map(lambda x: np.sum(np.array(x) >= 2015))
df_xbranch['BRANCH_3_OPEN_CNT'] = df_xbranch['B_REYEAR'].map(lambda x: np.sum(np.array(x) >= 2014))
df_xbranch['BRANCH_5_OPEN_CNT'] = df_xbranch['B_REYEAR'].map(lambda x: np.sum(np.array(x) >= 2012))

df_xbranch['BRANCH_1_CLOSE_CNT'] = df_xbranch['B_ENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2016))
df_xbranch['BRANCH_2_CLOSE_CNT'] = df_xbranch['B_ENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2015))
df_xbranch['BRANCH_3_CLOSE_CNT'] = df_xbranch['B_ENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2014))
df_xbranch['BRANCH_5_CLOSE_CNT'] = df_xbranch['B_ENDYEAR'].map(lambda x: np.sum(np.array(x) >= 2012))



# In[ ]:




# In[31]:

# df_xbranch.head()


# In[32]:

df_xbranch.to_csv('../data/public/3branch_1.csv',index=False,index_label=False)
# df_xbranch.columns


# In[33]:

df_all = pd.merge(df_all,df_xbranch[['EID', 'BRANCH_CNT','BRANCH_CNT_INHOME', 'BRANCH_CNT_OUTHOME', 'BRANCH_FIRST_YEAR',
                                    'BRANCH_END_YEAR', 'BRANCH_CLOSE_CNT', 'BRANCH_OPEN_CNT','BRANCH_CLOSE_RATE', 
                                    'BRANCH_OPEN_RATE', 'BRANCH_FIRST_CLOSE_YEAR','BRANCH_1_OPEN_CNT',
                                     'BRANCH_2_OPEN_CNT','BRANCH_3_OPEN_CNT','BRANCH_5_OPEN_CNT',
                                     'BRANCH_1_CLOSE_CNT','BRANCH_2_CLOSE_CNT','BRANCH_3_CLOSE_CNT','BRANCH_5_CLOSE_CNT']],how='left',on=['EID'])


# In[34]:

# df_all.head()


# In[35]:

# 空值填充，根据HY的类别的平均值或者众数进行填充
# ['BRANCH_CNT','BRANCH_CNT_INHOME', 'BRANCH_CNT_OUTHOME','BRANCH_CLOSE_CNT','BRANCH_OPEN_CNT', 'BRANCH_CLOSE_RATE', 'BRANCH_OPEN_RATE']使用同一个大类别的均值进行填充
# ['BRANCH_FIRST_YEAR','BRANCH_END_YEAR','BRANCH_FIRST_CLOSE_YEAR']使用同一个大类别的众数进行填充



HYLIST = set(df_all['HY'])

# print(HYLIST)

meanlist = ['BRANCH_CNT','BRANCH_CNT_INHOME', 'BRANCH_CNT_OUTHOME','BRANCH_CLOSE_CNT','BRANCH_OPEN_CNT', 'BRANCH_CLOSE_RATE', 'BRANCH_OPEN_RATE']
modelist = ['BRANCH_FIRST_YEAR','BRANCH_END_YEAR','BRANCH_FIRST_CLOSE_YEAR']


# In[36]:

# for HY in HYLIST:
#     # print(df_train['HY'].value_counts())
#     for d in meanlist:
#         df_all.loc[df_all[df_all[d].isnull()][df_all['HY']==HY].index,d] = df_all[df_all['HY']==HY][d].mean()
    
#     for c in modelist:
#         if(len(df_all[df_all['HY']==HY][c].value_counts().index)==0): continue
#         df_all.loc[df_all[df_all[c].isnull()][df_all['HY']==HY].index,c] = df_all[df_all['HY']==HY][c].value_counts().index[0]


# In[37]:

# 测试得到，当两个文件的重复的个数超过70%以上才能进行多合并，否则就进行根文件1的单合并

# df_xalter = pd.read_csv('../data/public/2alter_1.csv')
# df_tmp = df_xalter[df_xalter['EID'].isin(df_xbranch['EID'])]
# len(df_tmp)


# In[38]:

# len(df_xbranch)


# In[40]:

df_all[['BRANCH_1_OPEN_CNT','BRANCH_2_OPEN_CNT','BRANCH_3_OPEN_CNT','BRANCH_5_OPEN_CNT']] = df_all[['BRANCH_1_OPEN_CNT','BRANCH_2_OPEN_CNT','BRANCH_3_OPEN_CNT','BRANCH_5_OPEN_CNT']].fillna(0)

df_all[['BRANCH_1_CLOSE_CNT','BRANCH_2_CLOSE_CNT','BRANCH_3_CLOSE_CNT','BRANCH_5_CLOSE_CNT']] = df_all[['BRANCH_1_CLOSE_CNT','BRANCH_2_CLOSE_CNT','BRANCH_3_CLOSE_CNT','BRANCH_5_CLOSE_CNT']].fillna(0)




# In[41]:

df_all[meanlist] = df_all[meanlist].fillna(0)

# 所有关于年的空值使用注册时间填充

df_all.loc[df_all['BRANCH_FIRST_YEAR'].isnull(),'BRANCH_FIRST_YEAR'] = df_all.loc[df_all['BRANCH_FIRST_YEAR'].isnull(),'RGYEAR']
df_all.loc[df_all['BRANCH_END_YEAR'].isnull(),'BRANCH_END_YEAR'] = df_all.loc[df_all['BRANCH_END_YEAR'].isnull(),'RGYEAR']
df_all.loc[df_all['BRANCH_FIRST_CLOSE_YEAR'].isnull(),'BRANCH_FIRST_CLOSE_YEAR'] = df_all.loc[df_all['BRANCH_FIRST_CLOSE_YEAR'].isnull(),'RGYEAR']


# In[42]:

# df_all.head()


# In[43]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m
    
df_all['BRANCH_FIRST_YEAR_RGYEAR_DIFF'] = (df_all['BRANCH_FIRST_YEAR'] + ':' + df_all['RGYEAR']).apply(timeDiff)
df_all['BRANCH_END_YEAR_RGYEAR_DIFF'] = (df_all['BRANCH_END_YEAR'] + ':' + df_all['RGYEAR']).apply(timeDiff)

df_all['BRANCH_FIRST_CLOSE_YEAR_FIRST_DIFF'] = (df_all['BRANCH_FIRST_CLOSE_YEAR'] + ':' + df_all['BRANCH_FIRST_YEAR']).apply(timeDiff)
df_all['BRANCH_FIRST_CLOSE_YEAR_END_DIFF'] = (df_all['BRANCH_FIRST_CLOSE_YEAR'] + ':' + df_all['BRANCH_END_YEAR']).apply(timeDiff)



# In[44]:

df_all['PER_BRANCH_MONTH'] = df_all['BRANCH_END_YEAR_RGYEAR_DIFF'] / df_all['BRANCH_CNT']
df_all['PER_BRANCH_CLOSE_MONTH'] = df_all['BRANCH_END_YEAR_RGYEAR_DIFF'] / df_all['BRANCH_CLOSE_CNT']


#     14. 分支结构的数量占所有分支机构平均值的比例, BRANCH_CNT_ALL_RATE = BRANCH_CNT / 所有分支数据的平均值
#     15. 企业所在大类HY的对应的总分支机构的平均值，BRANCH_HY_AVG
#     16. 企业所在大类HY的对应的分支机构的平均值占所有机构的平均值的比例，BRANCH_HY_ALL_AVG_RATE
    
#     17. 分支结构的关停数量占所有关停分支机构平均值的比例, BRANCH_CLOSE_CNT_ALL_RATE = BRANCH_CLOSE_CNT / 所有关停分支数据的平均值
#     18. 企业所在大类HY的对应的关停分支机构的平均值，BRANCH_CLOSE_HY_AVG
#     19. 企业所在大类HY的对应的关停分支机构的平均值占所有关停机构的平均值的比例，BRANCH_HY_CLOSE_ALL_AVG_RATE
    
#     17. 分支结构的营业数量占所有营业分支机构平均值的比例, BRANCH_OPEN_CNT_ALL_RATE = BRANCH_OPEN_CNT / 所有营业分支数据的平均值
#     18. 企业所在大类HY的对应的营业分支机构的平均值，BRANCH_OPEN_HY_AVG
#     19. 企业所在大类HY的对应的营业分支机构的平均值占所有营业机构的平均值的比例，BRANCH_HY_OPEN_ALL_AVG_RATE
    
    
#     20. 平均几个月关闭一个分支结构，PER_BRANCH_CLOSE_MONTH =  BRANCH_END_YEAR_RGYEAR_DIFF / BRANCH_CLOSE_CNT 
    
# In[45]:

df_all['BRANCH_CNT_ALL_RATE'] = df_all['BRANCH_CNT'] / df_all['BRANCH_CNT'].mean()
df_all['BRANCH_CLOSE_CNT_ALL_RATE']  = df_all['BRANCH_CLOSE_CNT'] / df_all['BRANCH_CLOSE_CNT'].mean()
df_all['BRANCH_OPEN_CNT_ALL_RATE'] = df_all['BRANCH_OPEN_CNT'] / df_all['BRANCH_OPEN_CNT'].mean()


df_all['BRANCH_CNT_ALL_RATE_MAX'] = df_all['BRANCH_CNT'] / df_all['BRANCH_CNT'].max()
df_all['BRANCH_CLOSE_CNT_ALL_RATE_MAX']  = df_all['BRANCH_CLOSE_CNT'] / df_all['BRANCH_CLOSE_CNT'].max()
df_all['BRANCH_OPEN_CNT_ALL_RATE_MAX'] = df_all['BRANCH_OPEN_CNT'] / df_all['BRANCH_OPEN_CNT'].max()



# In[46]:

tmp = pd.DataFrame()

tmp['BRANCH_HY_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['BRANCH_CNT'].mean()
tmp['BRANCH_HY_ALL_AVG_RATE'] = tmp['BRANCH_HY_AVG'] / df_all['BRANCH_CNT'].mean()

tmp['BRANCH_CLOSE_HY_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['BRANCH_CLOSE_CNT'].mean()
tmp['BRANCH_HY_CLOSE_ALL_AVG_RATE'] = tmp['BRANCH_CLOSE_HY_AVG'] / df_all['BRANCH_CLOSE_CNT'].mean()

tmp['BRANCH_OPEN_HY_AVG'] = df_all.groupby(['HY'],as_index=True,axis=0)['BRANCH_OPEN_CNT'].mean()
tmp['BRANCH_HY_OPEN_ALL_AVG_RATE'] = tmp['BRANCH_OPEN_HY_AVG'] / df_all['BRANCH_OPEN_CNT'].mean()


tmp['BRANCH_HY_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['BRANCH_CNT'].max()
tmp['BRANCH_HY_ALL_MAX_RATE'] = tmp['BRANCH_HY_MAX'] / df_all['BRANCH_CNT'].max()

tmp['BRANCH_CLOSE_HY_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['BRANCH_CLOSE_CNT'].max()
tmp['BRANCH_HY_CLOSE_ALL_MAX_RATE'] = tmp['BRANCH_CLOSE_HY_MAX'] / df_all['BRANCH_CLOSE_CNT'].max()

tmp['BRANCH_OPEN_HY_MAX'] = df_all.groupby(['HY'],as_index=True,axis=0)['BRANCH_OPEN_CNT'].max()
tmp['BRANCH_HY_OPEN_ALL_MAX_RATE'] = tmp['BRANCH_OPEN_HY_MAX'] / df_all['BRANCH_OPEN_CNT'].max()


tmp['HY'] = tmp.index



# In[47]:

tmp1 = pd.DataFrame()

tmp1['BRANCH_ETYPE_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BRANCH_CNT'].mean()
tmp1['BRANCH_ETYPE_ALL_AVG_RATE'] = tmp1['BRANCH_ETYPE_AVG'] / df_all['BRANCH_CNT'].mean()

tmp1['BRANCH_CLOSE_ETYPE_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BRANCH_CLOSE_CNT'].mean()
tmp1['BRANCH_ETYPE_CLOSE_ALL_AVG_RATE'] = tmp1['BRANCH_CLOSE_ETYPE_AVG'] / df_all['BRANCH_CLOSE_CNT'].mean()

tmp1['BRANCH_OPEN_ETYPE_AVG'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BRANCH_OPEN_CNT'].mean()
tmp1['BRANCH_ETYPE_OPEN_ALL_AVG_RATE'] = tmp1['BRANCH_OPEN_ETYPE_AVG'] / df_all['BRANCH_OPEN_CNT'].mean()


tmp1['BRANCH_ETYPE_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BRANCH_CNT'].max()
tmp1['BRANCH_ETYPE_ALL_MAX_RATE'] = tmp1['BRANCH_ETYPE_MAX'] / df_all['BRANCH_CNT'].max()

tmp1['BRANCH_CLOSE_ETYPE_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BRANCH_CLOSE_CNT'].max()
tmp1['BRANCH_ETYPE_CLOSE_ALL_MAX_RATE'] = tmp1['BRANCH_CLOSE_ETYPE_MAX'] / df_all['BRANCH_CLOSE_CNT'].max()

tmp1['BRANCH_OPEN_ETYPE_MAX'] = df_all.groupby(['ETYPE'],as_index=True,axis=0)['BRANCH_OPEN_CNT'].max()
tmp1['BRANCH_ETYPE_OPEN_ALL_MAX_RATE'] = tmp1['BRANCH_OPEN_ETYPE_MAX'] / df_all['BRANCH_OPEN_CNT'].max()

tmp1['ETYPE'] = tmp1.index



# In[48]:

df_all = pd.merge(df_all,tmp,how='left',on=['HY'])
df_all = pd.merge(df_all,tmp1,how='left',on=['ETYPE'])



# In[49]:

# 00. 企业分支机构个数占对于大类HY的平均值的比例，BRANCH_HY_AVG_RATE
    
df_all['BRANCH_HY_AVG_RATE'] = df_all['BRANCH_CNT'] / df_all['BRANCH_HY_AVG']
df_all['BRANCH_HY_CLOSE_AVG_RATE'] = df_all['BRANCH_CLOSE_CNT'] / df_all['BRANCH_CLOSE_HY_AVG']
df_all['BRANCH_HY_OPEN_AVG_RATE'] = df_all['BRANCH_OPEN_CNT'] / df_all['BRANCH_OPEN_HY_AVG']

# 00. 企业分支机构个数占对于大类HY的平均值的比例，BRANCH_HY_MAX_RATE
df_all['BRANCH_HY_MAX_RATE'] = df_all['BRANCH_CNT'] / df_all['BRANCH_HY_MAX']
df_all['BRANCH_HY_CLOSE_MAX_RATE'] = df_all['BRANCH_CLOSE_CNT'] / df_all['BRANCH_CLOSE_HY_MAX']
df_all['BRANCH_HY_OPEN_MAX_RATE'] = df_all['BRANCH_OPEN_CNT'] / df_all['BRANCH_OPEN_HY_MAX']



# In[50]:


# 00. 企业分支机构个数占对于大类ETYPE的平均值的比例，BRANCH_ETYPE_AVG_RATE
    
df_all['BRANCH_ETYPE_AVG_RATE'] = df_all['BRANCH_CNT'] / df_all['BRANCH_ETYPE_AVG']
df_all['BRANCH_ETYPE_CLOSE_AVG_RATE'] = df_all['BRANCH_CLOSE_CNT'] / df_all['BRANCH_CLOSE_ETYPE_AVG']
df_all['BRANCH_ETYPE_OPEN_AVG_RATE'] = df_all['BRANCH_OPEN_CNT'] / df_all['BRANCH_OPEN_ETYPE_AVG']

# 00. 企业分支机构个数占对于大类ETYPE的平均值的比例，BRANCH_ETYPE_MAX_RATE
df_all['BRANCH_ETYPE_MAX_RATE'] = df_all['BRANCH_CNT'] / df_all['BRANCH_ETYPE_MAX']
df_all['BRANCH_ETYPE_CLOSE_MAX_RATE'] = df_all['BRANCH_CLOSE_CNT'] / df_all['BRANCH_CLOSE_ETYPE_MAX']
df_all['BRANCH_ETYPE_OPEN_MAX_RATE'] = df_all['BRANCH_OPEN_CNT'] / df_all['BRANCH_OPEN_ETYPE_MAX']




# In[51]:

# df_all.info()
# df_all.head()


# In[52]:

# 得到在df_xchange的所有数据，
df_all = df_all.fillna(0)

df_all.to_csv('../data/alldata/df_data123.csv',index=False,index_label=False)


# In[74]:

# In[ ]:



