
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# 重新审视数据，对年和月进行分割，对不听的类别的特征添加更过的one-hot编码
# ont-hot
# 1. HY
# 2. ETYPE
# 3. ALEREO
# 4. WXCODE
# 5. 年和月（日期）
# 
# 
# > 可以确定的one-hot编码:
# + HY
# + ETYPE
# + CHANGE_FIRST_ALTERNO
# + CHANGE_END_ALTERNO
# + CHANGE_ALTERNO_MUCNID
# + RIGHT_FIRST_TYPECODE
# + RIGHT_END_TYPECODE
# + RIGHT_TYPECODE_MUCHID
# 
# 删除只有一个特征的脏特征。
# 
# 
# 对所有的有关时间还能的特征，分成年和月两个维度，
# 
# timeType = ['RGYEAR','FIRST_CHANGE_TIME','END_CHANGE_TIME','BRANCH_FIRST_YEAR','BRANCH_END_YEAR',
#             'BRANCH_FIRST_CLOSE_YEAR','TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME',
#            'RIGHT_FIRST_ASK_TIME', 'RIGHT_FIRST_FB_TIME','RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME',
#            'PROJECT_FIRST_TIME', 'PROJECT_END_TIME', 'LAWSUIT_FIRST_TIME', 'LAWSUIT_END_TIME',            
#             'BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME','BREAKFAITH_END_FIRST_TIME',
#            'RECRUIT_FIRST_TIME', 'RECRUIT_END_TIME']
# 
# 继续添加特征，
# 增加注册资本 / 2018.01 - RGYEAR。
# 增加注册资本注册RATE / 2018.01 - RGYEAR。


# In[2]:

# 所有时间类型的特征
timeType = ['RGYEAR','FIRST_CHANGE_TIME','END_CHANGE_TIME','BRANCH_FIRST_YEAR','BRANCH_END_YEAR',
            'BRANCH_FIRST_CLOSE_YEAR','TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME',
           'RIGHT_FIRST_ASK_TIME', 'RIGHT_FIRST_FB_TIME','RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME',
           'PROJECT_FIRST_TIME', 'PROJECT_END_TIME', 'LAWSUIT_FIRST_TIME', 'LAWSUIT_END_TIME',            
            'BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME','BREAKFAITH_END_FIRST_TIME',
           'RECRUIT_FIRST_TIME', 'RECRUIT_END_TIME','QUALIFICATION_FIRST_FIRST_TIME', 
            'QUALIFICATION_FIRST_END_TIME','QUALIFICATION_END_FIRST_TIME' ]


# 所有进行one-hot的编码的数据,对注册时间和变成时间得到的年和月进行one-hot编码
onehot = ['PROV','HY','ETYPE','CHANGE_FIRST_ALTERNO','CHANGE_END_ALTERNO','CHANGE_ALTERNO_MUCNID',
          'RIGHT_FIRST_TYPECODE','RIGHT_END_TYPECODE','RIGHT_TYPECODE_MUCHID',
          'BRANCH_ETYPE_OPEN_ALL_MAX_RATE','BRANCH_OPEN_ETYPE_MAX','BREAKFAITH_2_OPEN_CNT',
          'BREAKFAITH_ETYPE_CNT_ALL_RATE_MAX','BREAKFAITH_ETYPE_CNT_MAX','CHANGE_ETYPE_CNT_ALL_MAX',
          'CHANGE_ETYPE_CNT_MAX','QUALIFICATION_1_CLOSE_CNT','QUALIFICATION_2_CLOSE_CNT',
          'QUALIFICATION_2_OPEN_CNT','QUALIFICATION_3_CLOSE_CNT','QUALIFICATION_3_OPEN_CNT',
          'QUALIFICATION_5_OPEN_CNT','QUALIFICATION_ETYPE_CNT_ALL_RATE_MAX','QUALIFICATION_ETYPE_CNT_MAX',
          'TZ_CLOSE_ETYPE_CNT_MAX','TZ_ETYPE_CLOSE_CNT_ALL_MAX_RATE','TZ_ETYPE_CNT_ALL_MAX_RATE',
          'TZ_ETYPE_CNT_MAX']

# 这些特征只有一个值，直接删除
oneValueFeature = ['BRANCH_1_OPEN_CNT','BREAKFAITH_1_OPEN_CNT','CHANGE_1_CNT',
                 'LAWSUIT_1_OPEN_CNT','PROJECT_1_OPEN_CNT','QUALIFICATION_1_OPEN_CNT',
                 'RECRUIT_1_OPEN_CNT','RIGHT_1_OPEN_CNT','TZ_1_OPEN_CNT',
                 'TZ_ETYPE_BTBL_ALL_MAX_RATE','TZ_ETYPE_BTBL_MAX']


# In[3]:

df_all = pd.read_csv('../data/alldata/df_data1234567890_plus.csv')


# In[4]:


# In[15]:

# df_all['RGYEAR_DIFF'].value_counts()


# In[5]:

x = list(df_all.columns)
x.sort()

# In[25]:

oneValueFeature = []

for f in x:
    if(len(set(df_all[f]))<=10 and len(set(df_all[f]))>2):
        print(f)
        oneValueFeature.append(f)


# In[ ]:




# In[30]:

# 会多出来50个特征，
k = 0
for feature in timeType:
#     print(feature)
    df_all[feature+'_YEAR'] =  df_all[feature].map(lambda x: int(x[:x.find('-')]))
    df_all[feature+'_MONTH'] = df_all[feature].map(lambda x: int(x[x.find('-')+1:]))
    print (k)
    k+=2


# In[32]:

# ont-hot编码，多出150个特征
for feature in onehot:
    df_all[feature] = df_all[feature].astype(np.int32)
    df_tmp = pd.get_dummies(df_all[feature], prefix=feature)
    df_all[df_tmp.columns] = df_tmp
    df_all.drop(feature,axis=1,inplace=True)
    print(feature)
    


# In[5]:

# 删除只有一个值的特征，这是脏特征
for f in oneValueFeature:
    df_all.drop(f,axis=1,inplace=True)


# In[6]:

# df_all.info()
# df_all.head()


# In[7]:

# 没有对时间进行one-hot编码
df_all.to_csv('../data/alldata/df_all_reviewed.csv',index=False,index_label=False)

# df_all.to_csv('../data/alldata/df_all1234567890_removeOneFeature.csv',index=False,index_label=False)


# In[ ]:



