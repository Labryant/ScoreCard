
# coding: utf-8

# In[2]:

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


import seaborn as sns
import matplotlib.pyplot as plt


# 目前最好的参数：
# ESP = 40
# max_depth = 7
# nround = 350
# 

# In[16]:

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
         'eta':0.05,
         'gamma ':0.1,
         'colsample_bytree':0.8, # old 0.8
         'subsample':0.8,
         'silent':1,
         'eval_metric':'auc',
         'objective':'binary:logistic',
#          'scale_pos_weight':6
        }


# In[4]:

df_all = pd.read_csv('../data/alldata/df_all1234567890_plus.csv')
df_all['XEID'] = df_all['EID'].map(lambda x: int(x[1:]))

# In[7]:

# 融合模型不需要运行
# 最后需要修改数据，不要在模型中调用数据的方式

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
    

# 默认填充的0，显示使用一个负数尝试一下
df_all.replace([np.inf, -np.inf], np.nan,inplace=True)
df_all = df_all.fillna(0)


# 设置特征数据，去除id数据，不能进行预测
features = df_all.columns[0:]
features = list(features)
features.remove('EID')
label = 'TARGET'

print("features的长度=",len(features))


# In[ ]:




# In[21]:

def split_build_valid(df_train,k=10):
    '''
    k-fold交叉验证,默认k=10
    df_train:训练数据
    '''

    #added some parameters
    kf = KFold(n_splits = k, shuffle = True, random_state = 400000)    
    result1 = next(kf.split(df_train[df_train[label]==1]), None)
    result0 = next(kf.split(df_train[df_train[label]==0]), None)
    
#     print (result)
#     (array([0, 2, 3, 5, 6, 7, 8, 9]), array([1, 4]))
    train_list = []
    train_list.extend(result1[0])
    train_list.extend(result0[0])
    np.random.shuffle(train_list)
    
    vali_list = []
    vali_list.extend(result1[1])
    vali_list.extend(result0[1])
    np.random.shuffle(vali_list)
    
    
#     dtrain = df_train.iloc[train_list]
#     dvali =  df_train.iloc[vali_list]
    
    dbuild = xgb.DMatrix(df_train.iloc[train_list][features],label=df_train.iloc[train_list][label])
    dvalid = xgb.DMatrix(df_train.iloc[vali_list][features],label=df_train.iloc[vali_list][label])
    watchlist = [(dbuild, 'build'),(dvalid, 'valid')]
    
    return dbuild, dvalid, watchlist

def split_train_valid(df_train,test_size=0.2):
    '''
    k-fold交叉验证,默认k=10
    df_train:训练数据
    '''
    X_train, X_vali, y_train, y_vali = train_test_split(df_train[features], df_train[label], test_size=test_size, random_state=40000)
    #added some parameters
    
#     dtrain = df_train.iloc[train_list]
#     dvali =  df_train.iloc[vali_list]
    
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dvalid = xgb.DMatrix(X_vali,label=y_vali)
    watchlist = [(dtrain, 'train'),(dvalid, 'valid')]
    
    return dtrain, dvalid, watchlist


# In[22]:

def xtrain_and_test(df_all):
    '''
    得到训练数据和测试数据
    '''
    df_label = pd.read_csv('../data/public/train.csv')
    df_test_label = pd.read_csv('../data/public/evaluation_public.csv')

    df_label.drop('ENDDATE',axis=1,inplace=True)
    
    df_train = df_all[df_all['EID'].isin(df_label['EID'])]
    df_test = df_all[df_all['EID'].isin(df_test_label['EID'])]
    
    df_train = pd.merge(df_train,df_label,how='left',on=['EID'])
    
    return df_train,df_test

df_train,df_test = xtrain_and_test(df_all)


# In[ ]:




# In[15]:

# print(len(df_train[df_train[label] == 1][label]))
# print(len(df_train[df_train[label] == 0][label]))


# In[23]:

# df_all.head()


# In[24]:

models = []
for i in range(LOOP):
    print('LOOP',i)
#     dbuild, dvalid, watchlist = split_build_valid(df_train)
    dbuild, dvalid, watchlist = split_train_valid(df_train,test_size=0.2)
            
        
    model = xgb.train(param, dbuild, nround, watchlist,early_stopping_rounds=ESR,verbose_eval=5)
    models.append(model)
    model.save_model('../saveModel/model1'+ str(i) + '.model')
    # VALID
    valid_yhat = model.predict(dvalid,ntree_limit=model.best_iteration)
    print('Valid Mean:---------------------->', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist


# In[25]:

# 得到最优参数，使用全部数据进行训练数据

models = []
for i in range(LOOP):
    print('LOOP',i)
    dbuild, dvalid, watchlist = split_train_valid(df_train,test_size=0.0001)
    nround = 400 # 在验证的基础上加上20轮左右
    model = xgb.train(param, dbuild,nround,watchlist,verbose_eval=5)
    models.append(model)
    model.save_model('../saveModel/model1234567'+ str(i) + '.model')
    # VALID
    valid_yhat = model.predict(dvalid,ntree_limit=model.best_iteration)
    print('Valid Mean:---------------------->', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist

#==============================================================================
print('test')
#==============================================================================

dtest  = xgb.DMatrix(df_test[features])
proba_test = pd.DataFrame()
proba_test['EID'] = df_test['EID']
proba_test['FORTARGET'] = [0 for i in range(len(df_test))]
proba_test['PROB'] = [0 for i in range(len(df_test))]
for model in models:
    proba_test['PROB'] += model.predict(dtest)
proba_test['PROB'] /= LOOP



# 计算特征重要程度
import operator
importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)


# 根据阈值设置标枪label的值
proba_test.loc[proba_test['PROB']>=0.23,'FORTARGET'] = 1


# 写入结果文件
proba_test.to_csv('../xresult/xsubmussion_plus.csv',index=False,index_label=False)


# In[30]:
print('预测的1的个数：',len(proba_test[proba_test['FORTARGET']==1]))
print('预测的0的个数：',len(proba_test[proba_test['FORTARGET']==0]))

