
# 根据xgboost的进行特征融合，进行训练和预测，得到训练的特征的初始代码（元代码）

# coding: utf-8

# In[4]:

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt


# In[33]:

#seed = np.random.randint(99999)
seed = 71

np.random.seed(seed)
valid_size = 0.2
LOOP = 1
ESR = 100
# XGB param
nround = 1500
#nround = 10

param = {'max_depth':7, # 基准是5 
         'learning_rate':0.05,
         'gamma ':0.1,
         'colsample_bytree':0.8, # old 0.8
         'subsample':0.8,
         'silent':1,
         'objective':'binary:logistic',
#          'scale_pos_weight':6
        }


# In[6]:

df_all = pd.read_csv('../data/alldata/df_data1234567890.csv')


# In[ ]:

# df_all['XEID'] = df_all['EID'].map(lambda x: int(x[1:]))



# In[9]:

timeType = ['RGYEAR','FIRST_CHANGE_TIME','END_CHANGE_TIME','BRANCH_FIRST_YEAR','BRANCH_END_YEAR',
            'BRANCH_FIRST_CLOSE_YEAR','TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME',
           'RIGHT_FIRST_ASK_TIME', 'RIGHT_FIRST_FB_TIME','RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME',
           'PROJECT_FIRST_TIME', 'PROJECT_END_TIME','LAWSUIT_FIRST_TIME', 'LAWSUIT_END_TIME',
           'BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME','BREAKFAITH_END_FIRST_TIME',
           'RECRUIT_FIRST_TIME','RECRUIT_END_TIME','QUALIFICATION_FIRST_FIRST_TIME',
           'QUALIFICATION_FIRST_END_TIME','QUALIFICATION_END_FIRST_TIME']


def time2int(x):
    '''
    将时间转化为整数，加入训练
    2018-01 - x
    '''
    y = int(x[:x.find('-')])
    m = int(x[x.find('-')+1:])
    s = 12*(2018-y) + (1-m)
    
    return s

# 尝试一下去电时间
for t in timeType:
    df_all[t] = df_all[t].apply(time2int)
    


# In[10]:

df_all.replace([np.inf, -np.inf], np.nan,inplace=True)
df_all = df_all.fillna(0)

# 默认填充的0，显示使用一个负数尝试一下


# In[20]:

features = df_all.columns[0:]
features = list(features)
features.remove('EID')
label = 'TARGET'

len(features)


# In[25]:

def xtrain_and_test(df_all):
    '''
    得到训练数据和测试数据
    '''
    df_label = pd.read_csv('../data/public/train.csv')
    df_test_label = pd.read_csv('../data/public/evaluation_public.csv')
#     print(len(df_label))
#     print(len(df_test_label))
    df_label.drop('ENDDATE',axis=1,inplace=True)
    
    df_train = df_all[df_all['EID'].isin(df_label['EID'])]
    df_test = df_all[df_all['EID'].isin(df_test_label['EID'])]
    
    df_train = pd.merge(df_train,df_label,how='left',on=['EID'])
    
    return df_train,df_test

df_train,df_test = xtrain_and_test(df_all)


# In[36]:

clf = XGBClassifier(max_depth=7,learning_rate=0.05,gamma=0.1,
                    colsample_bytree=0.8,subsample=0.8,
                    silent=1,objective='binary:logistic')

# clf = XGBClassifier(param)


# In[38]:

clf.fit(df_train[features],df_train[label],eval_metric='auc',verbose=5)



# In[44]:

prob = clf.predict_proba(df_test[features])


# In[ ]:




# In[45]:

proba_test = pd.DataFrame()
proba_test['EID'] = df_test['EID']
proba_test['FORTARGET'] = [0 for i in range(len(df_test))]
proba_test['PROB'] = prob[:,1]


# In[50]:

proba_test.loc[proba_test['PROB']>=0.23,'FORTARGET'] = 1


# In[51]:

proba_test.to_csv('../xresult/xsubmussion_1234567890_7_fit.csv',index=False,index_label=False)


# In[52]:

print(len(proba_test[proba_test['FORTARGET']==1]))
print(len(proba_test[proba_test['FORTARGET']==0]))



