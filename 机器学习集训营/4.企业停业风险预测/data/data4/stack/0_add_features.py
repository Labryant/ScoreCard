
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### 分析数据特征的目的，增加时间的差值的个数
# 1. 2alter.csv 和 5right.csv的数据较多，希望能高在这两个文件的融合出下下功夫
# 
# + (TZ,BRANCH,RECRUIT) ---> (CHANGE,RIGHT)
# + (LAWSUIT,PROJECT,QUALIFICATION,BREAKFAITH) ---> (RIGHT)
# + 左右两边就行了两两组合
# + 

# In[ ]:

timeType = ['RGYEAR','FIRST_CHANGE_TIME','END_CHANGE_TIME','BRANCH_FIRST_YEAR','BRANCH_END_YEAR',
            'BRANCH_FIRST_CLOSE_YEAR','TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME',
           'RIGHT_FIRST_ASK_TIME', 'RIGHT_FIRST_FB_TIME','RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME',
           'PROJECT_FIRST_TIME', 'PROJECT_END_TIME', 'LAWSUIT_FIRST_TIME', 'LAWSUIT_END_TIME',            
            'BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME','BREAKFAITH_END_FIRST_TIME',
           'RECRUIT_FIRST_TIME','RECRUIT_END_TIME','QUALIFICATION_FIRST_FIRST_TIME',
           'QUALIFICATION_FIRST_END_TIME','QUALIFICATION_END_FIRST_TIME']


# In[12]:

TBR = ['TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME','BRANCH_FIRST_YEAR','BRANCH_END_YEAR',
       'BRANCH_FIRST_CLOSE_YEAR','RECRUIT_FIRST_TIME','RECRUIT_END_TIME']

LPQ = ['PROJECT_FIRST_TIME', 'PROJECT_END_TIME','LAWSUIT_FIRST_TIME','LAWSUIT_END_TIME',
       'QUALIFICATION_FIRST_FIRST_TIME','QUALIFICATION_FIRST_END_TIME','QUALIFICATION_END_FIRST_TIME',
       'PROJECT_FIRST_TIME', 'PROJECT_END_TIME']

CR = ['FIRST_CHANGE_TIME','END_CHANGE_TIME','RIGHT_FIRST_ASK_TIME','RIGHT_END_ASK_TIME',]

R = ['RIGHT_FIRST_ASK_TIME','RIGHT_END_ASK_TIME']



# In[13]:

df_all = pd.read_csv('../data/alldata/df_data1234567890.csv')


# In[16]:

def timeDiff(x):
    a = x[:x.find(':')]
    b = x[x.find(':')+1:]
    y = int(a[:a.find('-')]) - int(b[:b.find('-')])
    m = int(a[a.find('-')+1:]) - int(b[b.find('-')+1:])
    return y * 12 + m

for f1 in TBR:
    for f2 in LPQ:
        df_all[f1+"_"+f2+"_DIFF"] = (df_all[f1] + ':' + df_all[f2]).apply(timeDiff)

for f1 in CR:
    for f2 in R:
        df_all[f1+"_"+f2+"_DIFF"] = (df_all[f1] + ':' + df_all[f2]).apply(timeDiff)

df_all.to_csv('../data/alldata/df_data1234567890_plus.csv',index=False,index_label=False)


# In[3]:



