
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier


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

def split_data_with_prov(df_data):
    '''
    根据特征PROV，分割数据，进行单独预测之后合并数据结果
    '''
    df_train_prov11 = df_data[df_data['PROV'] == 11]
    df_train_prov12 = df_data[df_data['PROV'] == 12]
    
    return df_train_prov11,df_train_prov12
    

def runXgbStack(inputfile, outputfile):
    '''
    输入输出文件，inputfile和outputfile
    '''
    # In[2]:

    df_all = pd.read_csv(inputfile)
    df_all['XEID'] = df_all['EID'].map(lambda x: int(x[1:]))

    # 默认填充的0，显示使用一个负数尝试一下
    df_all.replace([np.inf, -np.inf], np.nan,inplace=True)
    df_all = df_all.fillna(0)


    # 默认填充的0，显示使用一个负数尝试一下
    features = df_all.columns[0:]
    features = list(features)
    features.remove('EID')
    label = 'TARGET'
    
    clf = XGBClassifier(
        n_estimators=50,#50棵树
        learning_rate =0.05,
        max_depth=7,
        min_child_weight=1,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        seed=91)

    df_all_prov11,df_all_prov12 = split_data_with_prov(df_all)

    ###################### prov == 11
    df_train11,df_test11 = xtrain_and_test(df_all_prov11)
    # In[7]:
    

    X_train11 = df_train11[features]
    Y_label11 = df_train11[label]
    X_test11 = df_test11[features]

    clf.fit(X_train11,Y_label11,eval_metric='auc',verbose=5)
    column = ['STACKFEATURE'+str(i) for i in range(50)]
    df_new_feature11 = pd.DataFrame(clf.apply(df_all_prov11[features]),columns=column)
    df_all_prov11[column] = df_new_feature11

    ####################### prov == 12
    df_train12,df_test12 = xtrain_and_test(df_all_prov12)
    # In[7]:
    X_train12 = df_train12[features]
    Y_label12 = df_train12[label]
    X_test12 = df_test12[features]

    clf.fit(X_train12,Y_label12,eval_metric='auc',verbose=5)
    column = ['STACKFEATURE'+str(i) for i in range(50)]
    df_new_feature12 = pd.DataFrame(clf.apply(df_all_prov12[features]),columns=column)
    df_all_prov12[column] = df_new_feature12

    df_all = df_all_prov11.append(df_all_prov12)

    df_all.to_csv(outputfile,index=False,index_label=False)
    del df_all_prov11,df_all_prov12,df_all
    return outputfile

# run
inputfile = ['../data/alldata/df_data_all.csv','../data/alldata/df_data_onehot.csv',
             '../data/alldata/df_data_plus_all.csv','../data/alldata/df_data_plus_onehot.csv']

outputfile = ['../data/alldata/df_data_all_prov_xgbstack.csv','../data/alldata/df_data_onehot_prov_xgbstack.csv',
             '../data/alldata/df_data_plus_all_prov_xgbstack.csv','../data/alldata/df_data_plus_onehot_prov_xgbstack.csv']

# 本地测试
for i in range(0,1):
    print(i," start ",inputfile[i])
    runXgbStack(inputfile[i],outputfile[i])
    print(i," end ",outputfile[i])


