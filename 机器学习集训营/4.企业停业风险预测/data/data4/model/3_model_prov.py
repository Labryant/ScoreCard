
# coding: utf-8

# ## 根据省份数据，将所有的数据分成单个省进行训练和预测
# 
# + prov = 11
# + prov = 12
# 分成两个特征，这是目前最有的结果， 2017-11-27
# 
# In[1]:

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:

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


# In[3]:

df_all = pd.read_csv('../data/alldata/df_all1234567890_plus.csv')
df_all['XEID'] = df_all['EID'].map(lambda x: int(x[1:]))



# In[4]:

# 融合模型不需要运行

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
    


# In[5]:

# df_all['PROV'].value_counts()


# In[6]:

features = df_all.columns[0:]
features = list(features)
features.remove('EID')
features.remove('PROV')

label = 'TARGET'


# In[14]:

def split_build_valid(df_train,k=10):
    '''
    k-fold交叉验证,默认k=10
    df_train:训练数据
    '''

    #added some parameters
    kf = KFold(n_splits = k, shuffle = True, random_state = 400000)    
    result1 = next(kf.split(df_train[df_train[label]==1]), None)
    result0 = next(kf.split(df_train[df_train[label]==0]), None)
    
    train_list = []
    train_list.extend(result1[0])
    train_list.extend(result0[0])
    np.random.shuffle(train_list)
    
    vali_list = []
    vali_list.extend(result1[1])
    vali_list.extend(result0[1])
    np.random.shuffle(vali_list)
        
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

def split_data_with_prov(df_data):
    '''
    根据特征PROV，分割数据，进行单独预测之后合并数据结果
    '''
    df_train_prov11 = df_data[df_data['PROV'] == 11]
    df_train_prov12 = df_data[df_data['PROV'] == 12]
    
    return df_train_prov11,df_train_prov12
    


# In[8]:

df_train,df_test = xtrain_and_test(df_all)

# 根据省份11和12分成两份数据，进行预测。
df_train_prov11,df_train_prov12 = split_data_with_prov(df_train)
df_test_prov11,df_test_prov12 = split_data_with_prov(df_test)


# In[12]:

# print(len(df_train_prov11[df_train_prov11['TARGET']==1]))

# print(len(df_train_prov11[df_train_prov11['TARGET']==0]))


# In[15]:

print('PROV == 11')

models = []
for i in range(LOOP):
    print('11 LOOP',i)
#     dbuild, dvalid, watchlist = split_build_valid(df_train)
    dbuild, dvalid, watchlist = split_train_valid(df_train_prov11,test_size=0.2)
            
        
    model = xgb.train(param, dbuild, nround, watchlist,early_stopping_rounds=ESR,verbose_eval=5)
    models.append(model)
    model.save_model('../saveModel/model1'+ str(i) + '.model')
    # VALID
    valid_yhat = model.predict(dvalid,ntree_limit=model.best_iteration)
    print('11 Valid Mean:---------------------->', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist



# In[16]:

print('PROV == 12')
models = []
for i in range(LOOP):
    print('12 LOOP',i)
#     dbuild, dvalid, watchlist = split_build_valid(df_train)
    dbuild, dvalid, watchlist = split_train_valid(df_train_prov12,test_size=0.2)
            
        
    model = xgb.train(param, dbuild, nround, watchlist,early_stopping_rounds=ESR,verbose_eval=5)
    models.append(model)
    model.save_model('../saveModel/model1'+ str(i) + '.model')
    # VALID
    valid_yhat = model.predict(dvalid,ntree_limit=model.best_iteration)
    print('12 Valid Mean:---------------------->', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist



# ------------------------------------------------------------------------

# In[19]:

print('PROV == 11')

# 得到最优参数，使用全部数据进行训练数据
models = []
for i in range(LOOP):
    print('PROV 11 LOOP',i)
    dbuild, dvalid, watchlist = split_train_valid(df_train_prov11,test_size=0.001)
    nround = 250 # 在验证的基础上加上20轮左右
    model = xgb.train(param, dbuild,nround,watchlist,verbose_eval=5)
    models.append(model)
    model.save_model('../saveModel/model1234567'+ str(i) + '.model')
    # VALID
    valid_yhat = model.predict(dvalid,ntree_limit=model.best_iteration)
    print('PROV 11 Valid Mean:---------------------->', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist

#==============================================================================
print('PROV test 11')
#==============================================================================

dtest11  = xgb.DMatrix(df_test_prov11[features])
proba_test11 = pd.DataFrame()
proba_test11['EID'] = df_test_prov11['EID']
proba_test11['FORTARGET'] = [0 for i in range(len(df_test_prov11))]
proba_test11['PROB'] = [0 for i in range(len(df_test_prov11))]
for model in models:
    proba_test11['PROB'] += model.predict(dtest11)
proba_test11['PROB'] /= LOOP



# In[20]:

print('PROV == 12')

models = []
for i in range(LOOP):
    print('PROV 12 LOOP',i)
    dbuild, dvalid, watchlist = split_train_valid(df_train_prov12,test_size=0.001)
    nround = 250 # 在验证的基础上加上20轮左右
    model = xgb.train(param, dbuild,nround,watchlist,verbose_eval=5)
    models.append(model)
    model.save_model('../saveModel/model1234567'+ str(i) + '.model')
    # VALID
    valid_yhat = model.predict(dvalid,ntree_limit=model.best_iteration)
    print('12 Valid Mean:---------------------->', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist

#==============================================================================
print('PROV test 12')
#==============================================================================

dtest12  = xgb.DMatrix(df_test_prov12[features])
proba_test12 = pd.DataFrame()
proba_test12['EID'] = df_test_prov12['EID']
proba_test12['FORTARGET'] = [0 for i in range(len(df_test_prov12))]
proba_test12['PROB'] = [0 for i in range(len(df_test_prov12))]
for model in models:
    proba_test12['PROB'] += model.predict(dtest12)
proba_test12['PROB'] /= LOOP


# In[33]:

print(len(proba_test11[proba_test11['PROB']>=0.23]))
print(len(proba_test11[proba_test11['PROB']<=0.23]))


print(len(proba_test12[proba_test12['PROB']>=0.22]))
print(len(proba_test12[proba_test12['PROB']<=0.22]))


# In[34]:

proba_test11.loc[proba_test11['PROB']>=0.23,'FORTARGET'] = 1
proba_test12.loc[proba_test12['PROB']>=0.22,'FORTARGET'] = 1


# In[35]:

proba_tmp = proba_test11.append(proba_test12)



proba_test = pd.DataFrame()
proba_test['EID'] = df_test['EID']
proba_test['FORTARGET_TMP'] = [0 for i in range(len(df_test))]


# In[37]:

proba_test = pd.merge(proba_test,proba_tmp,how='left',on='EID')


# In[41]:

proba_test.drop('FORTARGET_TMP',axis=1,inplace=True)

proba_test.info()


# In[42]:

proba_test.to_csv('../xresult/xsubmussion_plus_prov.csv',index=False,index_label=False)


# In[43]:

print(len(proba_test[proba_test['FORTARGET']==1]))
print(len(proba_test[proba_test['FORTARGET']==0]))


