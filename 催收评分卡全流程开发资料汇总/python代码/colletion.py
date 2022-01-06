# -*- coding: utf-8 -*-
"""


@author: 番茄学院风控老骑士
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:47:27 2020

"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
from sklearn.preprocessing import StandardScaler

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

train = pd.read_csv(u'F:/trains.csv',encoding='gbk')



## 数值变量
numerical = [
'LOAN_AMOUNT',
'PERIOD_PERCEN',
'Inbound_Call',
'DPD_TIME_L1M',
'DPD_TIME_L2M',
'DPD_TIME_L3M',
'DPD_TIME_L4M',
'DPD_TIME_L5M',
'DPD_TIME_L6M',
'DPD_TIME_L7M',
'DPD_TIME_L8M',
'DPD_TIME_L9M',
'DPD_TIME_L10M',
'DPD_TIME_L11M',
'DPD_TIME_L12M',
'DPD_TIME_SUM',
'KPTP_L1M',
'KPTP_L2M',
'KPTP_L3M',
'KPTP_L4M',
'KPTP_L5M',
'KPTP_L6M',
'KPTP_L7M',
'KPTP_L8M',
'KPTP_L9M',
'KPTP_L10M',
'KPTP_L11M',
'KPTP_L12M',
'KPTP_SUM',
'BP_L1M',
'BP_L2M',
'BP_L3M',
'BP_L4M',
'BP_L5M',
'BP_L6M',
'BP_L7M',
'BP_L8M',
'BP_L9M',
'BP_L10M',
'BP_L11M',
'BP_L12M',
'BP_SUM',
'PTP_L1M',
'PTP_L2M',
'PTP_L3M',
'PTP_L4M',
'PTP_L5M',
'PTP_L6M',
'PTP_L7M',
'PTP_L8M',
'PTP_L9M',
'PTP_L10M',
'PTP_L11M',
'PTP_L12M',
'PTP_SUM',
'KPTP_RATE_L1M',
'KPTP_RATE_L2M',
'KPTP_RATE_L3M',
'KPTP_RATE_L4M',
'KPTP_RATE_L5M',
'KPTP_RATE_L6M',
'KPTP_RATE_L7M',
'KPTP_RATE_L8M',
'KPTP_RATE_L9M',
'KPTP_RATE_L10M',
'KPTP_RATE_L11M',
'KPTP_RATE_L12M',
'KPTP_RATE_SUM',
'DPD_L1M',
'DPD_L2M',
'DPD_L3M',
'DPD_L4M',
'DPD_L5M',
'DPD_L6M',
'DPD_L7M',
'DPD_L8M',
'DPD_L9M',
'DPD_L10M',
'DPD_L11M',
'DPD_L12M',
'DPD_SUM',
'DPD_L3M_SUM',
'DPD_L6M_SUM',
'DPD_L3M_MAX',
'DPD_L6M_MAX',
'DPD5_TIME',
'NOPAY_L1M',
'NOPAY_L2M',
'NOPAY_L3M',
'NOPAY_L4M',
'NOPAY_L5M',
'NOPAY_L6M',
'NOPAY_L7M',
'NOPAY_L8M',
'NOPAY_L9M',
'NOPAY_L10M',
'NOPAY_L11M',
'NOPAY_L12M',
'NOPAY_SUM',
'NOPAY_MAX'
]

target_var = ['target']

train_X = train[numerical]
train_Y = train[target_var]



## 对变量进行缺失值分析
missing = train_X.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

## 只有两列变量有值，本方法中不对缺失值做填充处理，用区别于原始分布的数值代替
train_X['KPTP_RATE_L1M'].fillna(-999, inplace=True)
train_X['KPTP_RATE_L2M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L3M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L4M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L5M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L6M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L7M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L8M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L9M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L10M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L11M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_L12M'].fillna(-999, inplace = True)
train_X['KPTP_RATE_SUM'].fillna(-999, inplace = True)
train_X['DPD_L3M_MAX'].fillna(-999, inplace = True)
train_X['DPD_L3M_SUM'].fillna(-999, inplace = True)
train_X['DPD_L6M_SUM'].fillna(-999, inplace = True)
train_X['DPD_L6M_MAX'].fillna(-999, inplace = True)




## 观察异常值


sns.boxplot(train['PERIOD_PERCEN'])


## 箱型图识别异常值标准
outile_var = [
'LOAN_AMOUNT',
'PERIOD_PERCEN',
'Inbound_Call',
'DPD_TIME_L1M',
'DPD_TIME_L2M',
'DPD_TIME_L3M',
'DPD_TIME_L4M',
'DPD_TIME_L5M',
'DPD_TIME_L6M',
'DPD_TIME_L7M',
'DPD_TIME_L8M',
'DPD_TIME_L9M',
'DPD_TIME_L10M',
'DPD_TIME_L11M',
'DPD_TIME_L12M',
'DPD_TIME_SUM',
'KPTP_L1M',
'KPTP_L2M',
'KPTP_L3M',
'KPTP_L4M',
'KPTP_L5M',
'KPTP_L6M',
'KPTP_L7M',
'KPTP_L8M',
'KPTP_L9M',
'KPTP_L10M',
'KPTP_L11M',
'KPTP_L12M',
'KPTP_SUM',
'BP_L1M',
'BP_L2M',
'BP_L3M',
'BP_L4M',
'BP_L5M',
'BP_L6M',
'BP_L7M',
'BP_L8M',
'BP_L9M',
'BP_L10M',
'BP_L11M',
'BP_L12M',
'BP_SUM',
'PTP_L1M',
'PTP_L2M',
'PTP_L3M',
'PTP_L4M',
'PTP_L5M',
'PTP_L6M',
'PTP_L7M',
'PTP_L8M',
'PTP_L9M',
'PTP_L10M',
'PTP_L11M',
'PTP_L12M',
'PTP_SUM',
'KPTP_RATE_L1M',
'KPTP_RATE_L2M',
'KPTP_RATE_L3M',
'KPTP_RATE_L4M',
'KPTP_RATE_L5M',
'KPTP_RATE_L6M',
'KPTP_RATE_L7M',
'KPTP_RATE_L8M',
'KPTP_RATE_L9M',
'KPTP_RATE_L10M',
'KPTP_RATE_L11M',
'KPTP_RATE_L12M',
'KPTP_RATE_SUM',
'DPD_L1M',
'DPD_L2M',
'DPD_L3M',
'DPD_L4M',
'DPD_L5M',
'DPD_L6M',
'DPD_L7M',
'DPD_L8M',
'DPD_L9M',
'DPD_L10M',
'DPD_L11M',
'DPD_L12M',
'DPD_SUM',
'DPD_L3M_SUM',
'DPD_L6M_SUM',
'DPD_L3M_MAX',
'DPD_L6M_MAX',
'DPD5_TIME',
'NOPAY_L1M',
'NOPAY_L2M',
'NOPAY_L3M',
'NOPAY_L4M',
'NOPAY_L5M',
'NOPAY_L6M',
'NOPAY_L7M',
'NOPAY_L8M',
'NOPAY_L9M',
'NOPAY_L10M',
'NOPAY_L11M',
'NOPAY_L12M',
'NOPAY_SUM',
'NOPAY_MAX'
]


### 用投票法来决定是否为异常值
for col in outile_var:
    qua_U = train[col].quantile(0.75)
    qua_L = train[col].quantile(0.25)
    IQR = qua_U - qua_L
    
    mean_values =  train[col].mean()
    std_values = train[col].std()
     
    qua_U1 = train[col].quantile(0.95)
    
    outile1 = qua_U + 1.5 * IQR
    outile2 = qua_U1
    outile3 = mean_values + 3 * std_values
    
    median = train[col].median()
    train_X.loc[((train_X[col] > outile1) & (train_X[col] > outile2) & (train_X[col] > outile3)), col] = median
    
    

## 自写卡方最优分箱过程
def get_chi2(X, col):
    '''
    计算卡方统计量
    '''
    # 计算样本期望频率
    
    pos_cnt = X['target'].sum()
    all_cnt = X['target'].count()
    expected_ratio = float(pos_cnt) / all_cnt 
    
    # 对变量按属性值从大到小排序
    df = X[[col, 'target']]
    col_value = list(set(df[col]))
    col_value.sort()

   # 计算每一个区间的卡方统计量
    
    chi_list = []
    pos_list = []
    expected_pos_list = []

    for value in col_value:
        df_pos_cnt = df.loc[df[col] == value, 'target'].sum()
        df_all_cnt = df.loc[df[col] == value,'target'].count()
        
        expected_pos_cnt = df_all_cnt * expected_ratio
        
        chi_square = (df_pos_cnt - expected_pos_cnt)**2 / expected_pos_cnt
        chi_list.append(chi_square)
        pos_list.append(df_pos_cnt)
        expected_pos_list.append(expected_pos_cnt)
    
    # 导出结果到dataframe
    chi_result = pd.DataFrame({col: col_value, 'chi_square':chi_list,
                               'pos_cnt':pos_list, 'expected_pos_cnt':expected_pos_list})

    return chi_result



def chiMerge(chi_result, maxInterval=5):
       
    '''
    根据最大区间数限制法则，进行区间合并
    '''
    
    group_cnt = len(chi_result)
    # 如果变量区间超过最大分箱限制，则根据合并原则进行合并，直至在maxInterval之内
    
    while(group_cnt > maxInterval):
        
        ## 取出卡方值最小的区间
        min_index = chi_result[chi_result['chi_square'] == chi_result['chi_square'].min()].index.tolist()[0]
        
        # 如果分箱区间在最前,则向下合并
        if min_index == 0:
            chi_result = merge_chiSquare(chi_result, min_index+1, min_index)
        
        # 如果分箱区间在最后，则向上合并
        elif min_index == group_cnt-1:
            chi_result = merge_chiSquare(chi_result, min_index-1, min_index)
        
        # 如果分箱区间在中间，则判断两边的卡方值，选择最小卡方进行合并
        else:
            if chi_result.loc[min_index-1, 'chi_square'] > chi_result.loc[min_index+1, 'chi_square']:
                chi_result = merge_chiSquare(chi_result, min_index, min_index+1)
            else:
                chi_result = merge_chiSquare(chi_result, min_index-1, min_index)
        
        group_cnt = len(chi_result)
    
    return chi_result



def merge_chiSquare(chi_result, index, mergeIndex, a = 'expected_pos_cnt',
                    b = 'pos_cnt', c = 'chi_square'):
    '''
    按index进行合并，并计算合并后的卡方值
    mergeindex 是合并后的序列值
    
    '''
    chi_result.loc[mergeIndex, a] = chi_result.loc[mergeIndex, a] + chi_result.loc[index, a]
    chi_result.loc[mergeIndex, b] = chi_result.loc[mergeIndex, b] + chi_result.loc[index, b]
    ## 两个区间合并后，新的chi2值如何计算
    chi_result.loc[mergeIndex, c] = (chi_result.loc[mergeIndex, b] - chi_result.loc[mergeIndex, a])**2 /chi_result.loc[mergeIndex, a]
    
    chi_result = chi_result.drop([index])
    
    ## 重置index
    chi_result = chi_result.reset_index(drop=True)
    
    return chi_result



## chi2分箱主流程
# 1：计算初始chi2 result
## 合并X数据集与Y数据集
train_all = pd.concat([train_X, train_Y], axis=1)

chi_result_all = dict()

for col in train_X.columns:
    print(col)
    print("start get " + col + " chi2 result")
    chi2_result = get_chi2(train_all, col)

    chi2_merge = chiMerge(chi2_result, maxInterval = 5)
    chi_result_all[col] = chi2_merge



## chi2分箱主流程
# 1：计算初始chi2 result
## 合并X数据集与Y数据集
 


chi2_merge['LOAN_AMOUNT']
chi2_merge['PERIOD_PERCEN']
chi2_merge['NOPAY_MAX']


chi_result_all['PERIOD_PERCEN']
chi_result_all['LOAN_AMOUNT']   


woe_iv={} ### 计算特征的IV值
IV={}


def get_woevalue(train_all, col, chi2_merge):
    ## 计算所有样本中，响应客户和未响应客户的比例
    df_pos_cnt = train_all['target'].sum()
    df_neg_cnt = train_all['target'].count() - df_pos_cnt    
    df_ratio = (df_pos_cnt / (df_neg_cnt * 1.0))
            
    col_interval = chi2_merge[col].values
    woe_list = []
    iv_list = []
    
    for i, val in enumerate(col_interval):
        if i == 0:
            col_pos_cnt = train_all.loc[train_all[col]<= val, 'target'].sum()
            col_all_cnt = train_all.loc[train_all[col]<= val, 'target'].count()
            col_neg_cnt = col_all_cnt - col_pos_cnt

        else:
            col_pos_cnt = train_all.loc[(train_all[col]<= val) & (train_all[col] > col_interval[i-1]), 'target'].sum()
            col_all_cnt = train_all.loc[(train_all[col]<= val) & (train_all[col] > col_interval[i-1]), 'target'].count()
            col_neg_cnt = col_all_cnt - col_pos_cnt
        
        if col_neg_cnt == 0:
            col_neg_cnt = col_neg_cnt + 1    
            
        if col_pos_cnt == 0:
            col_pos_cnt = col_pos_cnt + 1    
            
        col_ratio = col_pos_cnt / (col_neg_cnt * 1.0)
                
        woei = np.log(col_ratio / df_ratio)
        ivi = woei * ((col_pos_cnt / (df_pos_cnt * 1.0)) - (col_neg_cnt / (df_neg_cnt * 1.0)))
        woe_list.append(woei)
        iv_list.append(ivi)
    
    IV = sum(iv_list)
    
    return woe_list, iv_list, IV


        
for col in train_X.columns:
   
    ## 首先对特征进行分箱转化
    chi2_merge = chi_result_all[col]
    woe_list, iv_list, iv = get_woevalue(train, col, chi2_merge)
    woe_iv[col] = {'woe_list': woe_list, 'iv_list':iv_list, 'iv': iv, 'value_list':chi_result_all[col][col].values}



    

### 根据计算的IV值进行特征筛选
for col in train_X.columns:
    iv = woe_iv[col]['iv']
    if iv < 0.02:
        print(col)
        train_X.drop([col], axis=1) ## 删除IV值过小的特征

### 对留下的特征进行WOE编码转化,WOE编码只是为了使得评分卡的格式更加标准化，并不能提高模型的效果，分箱完过后，直接建立模型，一样可以达到目的

for col in train_X.columns:
    woe_list = woe_iv[col]['woe_list']
    col_interval = chi_result_all[col][col].values
    print(col)
    print(woe_list)
    print(col_interval)
    
    for i, val in enumerate(col_interval):
        if i == 0:
            train_X.loc[train_X[col] <= val, col] = woe_list[i]
        else:
            train_X.loc[(train_X[col] <= val) & (train_X[col] > col_interval[i-1]), col] = woe_list[i]
        



####对最终的数据集进行建模
####from sklearn.cross_validation import train_test_split
            
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split         
from sklearn.linear_model.logistic import LogisticRegression 
from sklearn.metrics import auc,roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score



### 切分训练集和测试集
X_train, X_test,y_train,y_test = train_test_split(train_X, train_Y, test_size=0.3)

c, r = y_train.shape
y_train = y_train.values.reshape(c,)

## 建立logistic回归模型 
lr = LogisticRegression(C=0.01)
lr.fit(X_train, y_train)

## 用拟合好的模型预测训练集
y_train_proba = lr.predict_proba(X_train)
y_train_label = lr.predict(X_train)

## 用拟合好的模型预测测试集
y_test_proba = lr.predict_proba(X_test)
y_test_label = lr.predict(X_test)


print('训练集准确率：{:.2%}'.format(accuracy_score(y_train, y_train_label)))
print('测试集准确率：{:.2%}'.format(accuracy_score(y_test, y_test_label)))

print('训练集精度：{:.2%}'.format(precision_score(y_train, y_train_label)))
print('测试集精度：{:.2%}'.format(precision_score(y_test, y_test_label)))

print('训练集召回率：{:.2%}'.format(recall_score(y_train, y_train_label)))
print('测试集召回率：{:.2%}'.format(recall_score(y_test, y_test_label)))

print('训练集AUC：{:.2%}'.format(roc_auc_score(y_train, y_train_proba[:,1])))
print('测试集AUC：{:.2%}'.format(roc_auc_score(y_test, y_test_proba[:,1])))
# ROC曲线和KS统计量
### ROC反映的是 TPR 与 FPR 之间的关系
### TPR = TP / (TP + FN) 灵敏度
### FPR = FP / (TN + FP) 误警率
### 绘制的是在不同阈值下的两者关系
### KS值小于0.2认为模型无鉴别能力

fpr, tpr, thresholds = roc_curve(y_test,y_test_proba[:,1], pos_label=1)
auc_score = auc(fpr,tpr)
w = tpr - fpr
ks_score = w.max()
ks_x = fpr[w.argmax()]
ks_y = tpr[w.argmax()]
fig,ax = plt.subplots()
ax.plot(fpr,tpr,label='AUC=%.5f'%auc_score)
ax.set_title('Receiver Operating Characteristic')
ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
ax.plot([ks_x,ks_x], [ks_x,ks_y], '--', color='red')
ax.text(ks_x,(ks_x+ks_y)/2,'  KS=%.5f'%ks_score)
ax.legend()
fig.show()          