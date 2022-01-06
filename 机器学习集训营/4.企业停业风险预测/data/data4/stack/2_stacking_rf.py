
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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


def runRfStack(inputfile,outputfile):
    '''
    输入输出文件
    '''
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

    df_train,df_test = xtrain_and_test(df_all)


    clf = RandomForestClassifier(
        n_estimators=50,#50棵树
        max_depth=7,
        n_jobs=4,
        random_state=101)

    X_train = df_train[features]
    Y_label = df_train[label]

    X_test = df_test[features]

    clf.fit(X_train,Y_label)
    column = ['STACKFEATURE'+str(i) for i in range(50)]
    df_new_feature = pd.DataFrame(clf.apply(df_all[features]),columns=column)
    df_all[column] = df_new_feature

    df_all.to_csv(outputfile,index=False,index_label=False)
    del df_train,df_test,df_all
    return outputfile
# In[ ]:

# run
inputfile = ['../data/alldata/df_data_all.csv','../data/alldata/df_data_onehot.csv',
             '../data/alldata/df_data_plus_all.csv','../data/alldata/df_data_plus_onehot.csv']

outputfile = ['../data/alldata/df_data_all_rfstack.csv','../data/alldata/df_data_onehot_rfstack.csv',
             '../data/alldata/df_data_plus_all_rfstack.csv','../data/alldata/df_data_plus_onehot_rfstack.csv']

for i in range(0,4):
    print(i," start ",inputfile[i])
    runRfStack(inputfile[i],outputfile[i])
    print(i," end ",inputfile[i])


