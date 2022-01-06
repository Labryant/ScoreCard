import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#seed = np.random.randint(99999)
valid_size = 0.2
LOOP = 3
ESR = 100
# XGB param
NROUND = 1500
nround = 300
features = None
lable = 'TARGET'


# 设置三组参数，使用不同的seed，分别是默认，27,9999，三个数，对单个测试文件的三个模型进行平均
param = [
    {'max_depth':7, # 基准是5 
        'eta':0.05,
        'gamma ':0.1,
        'colsample_bytree':0.8, # old 0.8
        'subsample':0.8,
        'silent':1,
        'eval_metric':'auc',
        'objective':'binary:logistic',
    },
    {'max_depth':7, # 基准是5 
        'eta':0.05,
        'gamma ':0.1,
        'colsample_bytree':0.8, # old 0.8
        'subsample':0.8,
        'silent':1,
        'eval_metric':'auc',
        'objective':'binary:logistic',
        'seed':27
    },
    {'max_depth':7, # 基准是5 
        'eta':0.05,
        'gamma ':0.1,
        'colsample_bytree':0.8, # old 0.8
        'subsample':0.8,
        'silent':1,
        'eval_metric':'auc',
        'objective':'binary:logistic',
        'seed':9999
    }
]

def setFeaturesAndLable(df_columns):
    '''
    参数：读入数据的df_data的所有列名
    得到数据的特征features和lable，
    这里所有的特征都是去除EID
    '''
    xfeatures = df_columns
    xfeatures = list(xfeatures)
    xfeatures.remove('EID')
    xlabel = 'TARGET'
    
    global features, label
    features, label = xfeatures,xlabel

    return xfeatures, xlabel


def split_train_valid(df_train,test_size=0.2):
    '''
    k-fold交叉验证,默认k=5
    df_train:训练数据
    '''
    X_train, X_vali, y_train, y_vali = train_test_split(df_train[features], df_train[label], test_size=test_size, random_state=40000)
        
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

def getBestIteration(df_train):
    '''
    参数：所有训练数据
    根据设置第一组参数和5折验证集合，返回得到最好的迭代次数，
    给nround赋值，下面使用全部训练数据，设置得带次数nround = best_iteration + 50
    '''

    dbuild, dvalid, watchlist = split_train_valid(df_train,test_size=0.2)        
    model = xgb.train(param[0], dbuild, NROUND, watchlist,early_stopping_rounds=ESR,verbose_eval=5)
    valid_yhat = model.predict(dvalid,ntree_limit=model.best_iteration)
    print('Valid Mean:---------------------->', np.mean(valid_yhat))
    del dbuild, dvalid, watchlist

    # 设置全局变量的值
    nround = model.best_iteration + 50

    return model.best_iteration + 50


def XgbModel(df_train,df_test,nround,alpha = 0.23):
    '''
    参数：训练数据和测试数据
        outputfile：输出文件的结果。
        alpha设置是1的阈值，默认是0.23
    return 数据文件的结果，三个不同seed的平均是的
    ''' 
    models = []
    for i in range(LOOP):
        print('LOOP',i)
        dbuild, dvalid, watchlist = split_train_valid(df_train,test_size=0.1)

        model = xgb.train(param[i], dbuild,nround,watchlist,verbose_eval=5)
        models.append(model)
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

    # 根据阈值设置标枪label的值
    proba_test.loc[proba_test['PROB']>=alpha,'FORTARGET'] = 1
    # 写入结果文件
    # proba_test.to_csv(output_file,index=False,index_label=False)

    return proba_test


def runModel(inputfile,outputfile):
    """
    inputfiles输入文件，
    """
    df_all = pd.read_csv(inputfile)
    # 设置特征和lable
    setFeaturesAndLable(df_all.columns)

    # 
    df_train,df_test = xtrain_and_test(df_all)

    nround = getBestIteration(df_train)
    
    proba_test = XgbModel(df_train,df_test,nround)
    
    proba_test.to_csv(outputfile,index=False,index_label=False)

    # 释放空间
    del df_train,df_test, df_all
    return proba_test

inputfile = ["../data/alldata/df_data_all.csv","../data/alldata/df_data_plus_all.csv",
             "../data/alldata/df_data_onehot.csv","../data/alldata/df_data_plus_onehot.csv",
             '../data/alldata/df_data_all_xgbstack.csv','../data/alldata/df_data_plus_all_xgbstack.csv',
             '../data/alldata/df_data_onehot_xgbstack.csv','../data/alldata/df_data_plus_onehot_xgbstack.csv',
             '../data/alldata/df_data_all_rfstack.csv','../data/alldata/df_data_plus_all_rfstack.csv',
             '../data/alldata/df_data_onehot_rfstack.csv','../data/alldata/df_data_plus_onehot_rfstack.csv',
             '../data/alldata/df_data_all_lgbstack.csv','../data/alldata/df_data_plus_all_lgbstack.csv',
             '../data/alldata/df_data_onehot_lgbstack.csv','../data/alldata/df_data_plus_onehot_lgbstack.csv',
             '../data/alldata/df_data_all_prov_xgbstack.csv','../data/alldata/df_data_plus_all_prov_xgbstack.csv',
             '../data/alldata/df_data_onehot_prov_xgbstack.csv','../data/alldata/df_data_plus_onehot_prov_xgbstack.csv',
             '../data/alldata/df_data_all_prov_rfstack.csv','../data/alldata/df_data_plus_all_prov_rfstack.csv',
             '../data/alldata/df_data_onehot_prov_rfstack.csv','../data/alldata/df_data_plus_onehot_prov_rfstack.csv',
             '../data/alldata/df_data_all_prov_lgbstack.csv','../data/alldata/df_data_plus_all_prov_lgbstack.csv',
             '../data/alldata/df_data_onehot_prov_lgbstack.csv','../data/alldata/df_data_plus_onehot_prov_lgbstack.csv',
             '../data/alldata/df_data_all_xgbstack_onehot.csv','../data/alldata/df_data_onehot_xgbstack_onehot.csv',
             '../data/alldata/df_data_plus_all_xgbstack_onehot.csv','../data/alldata/df_data_plus_onehot_xgbstack_onehot.csv']
             
outputfile = ['../xresult/sub_all.csv','../xresult/sub_plus_all.csv',
              '../xresult/sub_onehot.csv','../xresult/sub_plus_onehot.csv',
              '../xresult/sub_all_xgbstack.csv','../xresult/sub_plus_all_xgbstack.csv',
              '../xresult/sub_onehot_xgbstack.csv','../xresult/sub_plus_onehot_xgbstack.csv',
              '../xresult/sub_all_rfstack.csv','../xresult/sub_plus_all_rfstack.csv',
              '../xresult/sub_onehot_rfstack.csv','../xresult/sub_plus_onehot_rfstack.csv',
              '../xresult/sub_all_lgbstack.csv','../xresult/sub_plus_all_lgbstack.csv',
              '../xresult/sub_onehot_lgbstack.csv','../xresult/sub_plus_onehot_lgbstack.csv',
              '../xresult/sub_all_prov_xgbstack.csv','../xresult/sub_plus_all_prov_xgbstack.csv',
              '../xresult/sub_onehot_prov_xgbstack.csv','../xresult/sub_plus_onehot_prov_xgbstack.csv',
              '../xresult/sub_all_prov_rfstack.csv','../xresult/sub_plus_all_prov_rfstack.csv',
              '../xresult/sub_onehot_prov_rfstack.csv','../xresult/sub_plus_onehot_prov_rfstack.csv',
              '../xresult/sub_all_prov_lgbstack.csv','../xresult/sub_plus_all_prov_lgbstack.csv',
              '../xresult/sub_onehot_prov_lgbstack.csv','../xresult/sub_plus_onehot_prov_lgbstack.csv',
              '../xresult/a.csv','../xresult/b.csv',
              '../xresult/c.csv','../xresult/d.csv']


# inputfile = "../data/alldata/df_data_all.csv"
# outputfile = '../xresult/sub_all.csv'

# 28个文件
for i in range(16,28):
    print(i," start --> ",inputfile[i])
    runModel(inputfile[i],outputfile[i])
    print(i," end --> ",inputfile[i])
                                                                                
