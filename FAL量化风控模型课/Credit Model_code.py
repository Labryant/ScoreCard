# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:23:38 2019

@author: Administrator
"""

## 工作包准备，numpy和pandas是常用的数据分析第三方包
import numpy as np
import pandas as pd 
from scipy.stats import chi2

## 利用pandas自带的read_csv导入数据，导入的数据会转化为pandas数据格式，dataframe类型
train = pd.read_csv('C:/Users/Administrator/Desktop/MicroSynFinal.csv')

#### 对数据集进行描述性统计分析 ###

numerical = ['Collateral_valuation',
             'Age',
             'Properties_Total',
             'Amount',
             'Term',
             'Historic_Loans',
             'Current_Loans',
             'Max_Arrears']

categorical = ['Region',
               'Area',
               'Activity',
               'Properties_Status']

binaray = ['Guarantor',
           'Collateral']

### 将目标变量单独赋值给一个变量
target_var = ['Defaulter']

train_X = train[numerical + categorical + binaray]
train_Y = train[target_var]

train_X.describe()

### 首先将类别变量转换为虚拟变量，方便之后做数据探索
dummy_region = pd.get_dummies(train_X["Region"],prefix='Region')
dummy_region_col = list(dummy_region.columns)
dummy_area = pd.get_dummies(train_X["Area"],prefix='Area')
dummy_area_col = list(dummy_area.columns)
dummy_activity = pd.get_dummies(train_X["Activity"],prefix='Activity', dummy_na=True)
dummy_activity_col = list(dummy_activity.columns)
dummy_status = pd.get_dummies(train_X["Properties_Status"],prefix='PropertiesStatus')
dummy_status_col = list(dummy_status.columns)
dummy_col_dict = {"Region":dummy_region_col, "Area":dummy_area_col, "Activity":dummy_activity_col, "Properties_Status":dummy_status_col}

### 分别取自变量数据集和目标变量数据集
train_X = pd.concat([train[numerical+binaray],dummy_region, dummy_area, dummy_activity, dummy_status], axis=1)
train_Y = train[target_var]

train = pd.concat([train_X, train_Y], axis=1)

### 对数据集做描述性分析
train_X.describe()

### 基于target变量，分别进行describe
train[train['Defaulter']==0].describe()
train[train['Defaulter']==1].describe()

## 数据探索--协方差和相关矩阵

train.cov()
train.corr()

### 绘制直方图和箱形图

from matplotlib import pyplot as plt
plt.hist(train[train['Defaulter']==0]['Age'],color='blue',label='Class 0',alpha=0.5,bins=20)
plt.hist(train[train['Defaulter']==1]['Age'],color='red',label='Class 1',alpha=0.5,bins=20)

plt.legend(loc='best')
plt.grid()
plt.show()

train[['Defaulter', 'Age']].boxplot(by='Defaulter',layout=(1,1))
plt.show()


## 首先做缺失值处理
missing = train_X.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

## 两列缺失值，一列是当前在还贷款总额，一列是抵押物价值，从数据看出，抵押物价值为空，就是没有抵押物的意思，已有是否有抵押物表示，这列变量不用对空值处理
### 一列是当前在还贷款总额，如果为空，则表示当前没有在还贷款，遵循空值即信息的原则
train_X.loc[train_X['Current_Loans'].isnull(), 'Current_Loans_nan'] = 1
train_X.loc[train_X['Current_Loans_nan'].isnull(), 'Current_Loans_nan'] = 0

binaray = binaray + ["Current_Loans_nan"]


train = pd.concat([train_X, train_Y], axis=1)

### 将方差较小的变量，直接选择进行剔除,阈值选择0.001 ##
## 针对数值变量做方差筛选
drop_col = list()
for col in numerical:
    col_var = train_X[col].var()
    if col_var < 0.001:
        drop_col.append(col)
        train_X.drop(axis=1, columns=col, inplace=True)

numerical = list(set(numerical).difference(set(drop_col)))
train = pd.concat([train_X, train_Y], axis=1)

### 缺失值处理完成过后，如果样本比例不均匀，则进行样本调整，本例子的样本比例在1:5,因此可以不用对样本比例进行调整##
### 统计目标变量好样本和坏样本的个数
'''
neg_Y = train_Y.sum()
pos_Y = train_Y.count() - neg_Y

### 好坏样本的比例差距过大，我们采用分层抽样的方法，对样本比例做调整
### 将数据集好坏样本进行区分，P_train为好样本数据集，N_train为坏样本数据集
P_train = train[train['Defaulter'] == 0]
N_train = train[train['Defaulter'] == 1]

### 对好样本进行抽样，抽样个数选择坏样本个数的5倍
P_train_sample = P_train.sample(n=N_train.shape[0] * 5, frac=None, replace=False, weights=None, random_state=2, axis=0)
print P_train_sample.shape
print N_train.shape

### 将抽样的好样本数据集与坏样本数据集合并，重新生成训练集
train_sample = pd.concat([N_train,P_train_sample])
print train_sample.shape

### 将新训练集的index进行重排
train_sample= train_sample.sample(frac=1).reset_index(drop=True)
'''


## 自写卡方最优分箱过程
def get_chi2(X, col):
    '''
    计算卡方统计量
    '''
    # 计算样本期望频率
    
    pos_cnt = X['Defaulter'].sum()
    all_cnt = X['Defaulter'].count()
    expected_ratio = float(pos_cnt) / all_cnt 
    
    # 对变量按属性值从大到小排序
    df = X[[col, 'Defaulter']]
    df = df.dropna()
    col_value = list(set(df[col]))
    col_value.sort()
    
    # 计算每一个区间的卡方统计量
    
    chi_list = []
    pos_list = []
    expected_pos_list = []
    
    for value in col_value:
        df_pos_cnt = df.loc[df[col] == value, 'Defaulter'].sum()
        df_all_cnt = df.loc[df[col] == value,'Defaulter'].count()
        
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


def cal_chisqure_threshold(dfree=4, cf=0.1):
    '''
    根据给定的自由度和显著性水平, 计算卡方阈值
    '''
    percents = [0.95, 0.90, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
    
    ## 计算每个自由度，在每个显著性水平下的卡方阈值
    df = pd.DataFrame(np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))
    df.columns = percents
    df.index = df.index+1
    
    pd.set_option('precision', 3)
    return df.loc[dfree, cf]


def chiMerge_chisqure(chi_result, dfree=4, cf=0.1, maxInterval=5):

    threshold = cal_chisqure_threshold(dfree, cf)
    
    min_chiSquare = chi_result['chi_square'].min()
    
    group_cnt = len(chi_result)
    
    
    # 如果变量区间的最小卡方值小于阈值，则继续合并直到最小值大于等于阈值
    
    while(min_chiSquare < threshold and group_cnt > maxInterval):
        min_index = chi_result[chi_result['chi_square']==chi_result['chi_square'].min()].index.tolist()[0]
        
        # 如果分箱区间在最前,则向下合并
        if min_index == 0:
            chi_result = merge_chiSquare(chi_result, min_index+1, min_index)
        
        # 如果分箱区间在最后，则向上合并
        elif min_index == group_cnt-1:
            chi_result = merge_chiSquare(chi_result, min_index-1, min_index)
        
        # 如果分箱区间在中间，则判断与其相邻的最小卡方的区间，然后进行合并
        else:
            if chi_result.loc[min_index-1, 'chi_square'] > chi_result.loc[min_index+1, 'chi_square']:
                chi_result = merge_chiSquare(chi_result, min_index, min_index+1)
            else:
                chi_result = merge_chiSquare(chi_result, min_index-1, min_index)
        
        min_chiSquare = chi_result['chi_square'].min()
        
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

### 先对数据进行等频分箱，提高卡方分箱的效率

## 注意对原始数据的拷贝
import copy
chi_train_X = copy.deepcopy(train_X)

### 本例先不进行等频分箱的过程
'''
def get_freq(train_X, col, bind):
    col_data = train_X[col]
    col_data_sort = col_data.sort_values().reset_index(drop=True)
    col_data_cnt = col_data.count()
    length = col_data_cnt / bind
    col_index = np.append(np.arange(length, col_data_cnt, length), (col_data_cnt - 1))
    col_interval = list(set(col_data_sort[col_index]))
    return col_interval    
'''    

'''  
for col in train_X.columns:
    print "start get " + col + " 等频 result"
    col_interval = get_freq(train_X, col, 200)
    col_interval.sort()
    for i, val in enumerate(col_interval):
        if i == 0:
            freq_train_X.loc[train_X[col] <= val, col] = i + 1 
            
        else:
            freq_train_X.loc[(train_X[col]<= val) & (train_X[col] > col_interval[i-1]), col] = i + 1
        
'''    

## 对数据进行卡方分箱，按照自由度进行分箱

chi_result_all = dict()

for col in chi_train_X.columns:
    print("start get " + col + " chi2 result")
    chi2_result = get_chi2(train, col)
    chi2_merge = chiMerge_chisqure(chi2_result, dfree=4, cf=0.05, maxInterval=5)
    
    chi_result_all[col] = chi2_merge

 
### 进行WOE编码

woe_iv={} ### 计算特征的IV值

def get_woevalue(train_all, col, chi2_merge):
    ## 计算所有样本中，响应客户和未响应客户的比例
    df_pos_cnt = train_all['Defaulter'].sum()
    df_neg_cnt = train_all['Defaulter'].count() - df_pos_cnt
    
    df_ratio = df_pos_cnt / (df_neg_cnt * 1.0)
    
        
    col_interval = chi2_merge[col].values
    woe_list = []
    iv_list = []
    
    for i, val in enumerate(col_interval):
        if i == 0:
            col_pos_cnt = train_all.loc[train_all[col]<= val, 'Defaulter'].sum()
            col_all_cnt = train_all.loc[train_all[col]<= val, 'Defaulter'].count()
            col_neg_cnt = col_all_cnt - col_pos_cnt
        
        else:
            col_pos_cnt = train_all.loc[(train_all[col]<= val) & (train_all[col] > col_interval[i-1]), 'Defaulter'].sum()
            col_all_cnt = train_all.loc[(train_all[col]<= val) & (train_all[col] > col_interval[i-1]), 'Defaulter'].count()
            col_neg_cnt = col_all_cnt - col_pos_cnt
        
        if col_neg_cnt == 0:
            col_neg_cnt = col_neg_cnt + 1
        
        col_ratio = col_pos_cnt / (col_neg_cnt * 1.0)
        
        
        woei = np.log(col_ratio / df_ratio)
        ivi = woei * ((col_pos_cnt / (df_pos_cnt * 1.0)) - (col_neg_cnt / (df_neg_cnt * 1.0)))
        woe_list.append(woei)
        iv_list.append(ivi)
    
    IV = sum(iv_list)
    
    return woe_list, iv_list, IV
        
        
for col in chi_train_X.columns:
    
    ## 首先对特征进行分箱转化
    chi2_merge = chi_result_all[col]
    woe_list, iv_list, iv = get_woevalue(train, col, chi2_merge)
    woe_iv[col] = {'woe_list': woe_list, 'iv_list':iv_list, 'iv': iv, 'value_list':chi_result_all[col][col].values}

### 计算字符变量的总体iv值
              
woe_iv['Region'] = {'woe_list':[woe_iv[col]['woe_list'][1] for col in dummy_region_col], 'iv': np.sum([woe_iv[col]['iv_list'][1] for col in dummy_region_col]),'value_list':[col.split('_')[1] for col in dummy_region_col]}
woe_iv['Area'] = {'woe_list':[woe_iv[col]['woe_list'][1] for col in dummy_area_col], 'iv': np.sum([woe_iv[col]['iv_list'][1] for col in dummy_area_col]),'value_list':[col.split('_')[1] for col in dummy_area_col]}
woe_iv['Activity'] = {'woe_list':[woe_iv[col]['woe_list'][1] for col in dummy_activity_col], 'iv': np.sum([woe_iv[col]['iv_list'][1] for col in dummy_activity_col]), 'value_list': [col.split('_')[1] for col in dummy_activity_col]}
woe_iv['Properties_Status'] = {'woe_list':[woe_iv[col]['woe_list'][1] for col in dummy_status_col], 'iv': np.sum([woe_iv[col]['iv_list'][1] for col in dummy_status_col]), 'value_list':[col.split('_')[1] for col in dummy_status_col]}

   
### 根据计算的IV值进行特征筛选
drop_numerical = list()
for col in numerical:
    iv = woe_iv[col]['iv']
    if iv < 0.02:
        drop_numerical.append(col)
        chi_train_X.drop(axis=1, columns=col, inplace=True) ## 删除IV值过小的特征

drop_categorical = list()
for col in categorical:
    iv = woe_iv[col]['iv']
    if iv < 0.02:
        drop_categorical.append(col)
        chi_train_X.drop(axis=1, columns=dummy_col_dict[col], inplace=True)

drop_binary = list()
for col in binaray:
    iv = woe_iv[col]['iv']
    if iv < 0.02:
        drop_binary.append(col)
        chi_train_X.drop(axis=1, columns=col, inplace=True)



numerical = list(set(numerical).difference(drop_numerical))
categorical = list(set(categorical).difference(drop_categorical))
binaray = list(set(binaray).difference(drop_binary))
### 对留下的特征进行WOE编码转化,WOE编码只是为了使得评分卡的格式更加标准化，并不能提高模型的效果，分箱完过后，直接建立模型，一样可以达到目的

woe_train_X = copy.deepcopy(chi_train_X)

for col in numerical:
    woe_list = woe_iv[col]['woe_list']
    col_interval = woe_iv[col]['value_list']
    
    for i, val in enumerate(col_interval):
        if i == 0:
            woe_train_X.loc[chi_train_X[col] <= val, col] = woe_list[i]
        else:
            woe_train_X.loc[(chi_train_X[col] <= val) & (chi_train_X[col] > col_interval[i-1]), col] = woe_list[i]
    woe_train_X.loc[woe_train_X[col].isnull(), col] = 0

for col in categorical:
    woe_list = woe_iv[col]['woe_list']
    col_interval = woe_iv[col]['value_list']
    
    for i, val in enumerate(col_interval):
        woe_train_X.loc[woe_train_X[dummy_col_dict[col][i]]==1 , col] = woe_list[i]
    woe_train_X.drop(axis=1, columns=dummy_col_dict[col], inplace=True)


for col in binaray:
    woe_list = woe_iv[col]['woe_list']
    col_interval = woe_iv[col]['value_list']
    
    for i,var in enumerate(col_interval):
        woe_train_X.loc[woe_train_X[col]==var , col] = woe_list[i]
    
    

### 在数据集中加上intercept列
woe_train_X['intercept'] = [1] * woe_train_X.shape[0]

train_all = pd.concat([woe_train_X, train_Y], axis=1)


### 将数据集进行切分，以便后续对模型做验证
from sklearn.model_selection import train_test_split

### 切分训练集和测试集,按照7:3的比例进行切分
train_all_train, train_all_test = train_test_split(train_all, test_size=0.3)


import statsmodels.formula.api as smf
import pandas as pd
 

def forward_selected(train_data, target):
    
    remaining = set(train_data.columns)
    remaining.remove(target)
    remaining.remove('intercept')
    
    selected = ['intercept']
    current_score, best_new_score = float("inf"),float("inf") 
    
    while remaining and current_score == best_new_score:
        scores_candidates = []
        for candidate in remaining:
            #formula = "{} ~ {} + 1".format(target,  ' + '.join(selected + [candidate]))
            score = smf.Logit(train_data[target], train_data[selected + [candidate]] ).fit().bic
            #score = smf.logit(formula, train_data).fit().bic
            
            scores_candidates.append((score, candidate))
            
        scores_candidates.sort(reverse = True)
        print(scores_candidates)
        
        best_new_score, best_candidate = scores_candidates.pop()
        
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    
    #formula = "{} ~ {} + 1".format(target, ' + '.join(selected))
    model = smf.Logit(train_data[target], train_data[selected]).fit()
 
    return model
  
    
model = forward_selected(train_all_train, 'Defaulter')

print(model.params)
    
print(model.bic)
 
##### 对模型中的每个变量做wald 卡方检验
for col in model.params.index:
    result = model.wald_test(col)
    print(str(col) + " wald test: " + str(result.pvalue))


### 查看VIF值
from statsmodels.stats.outliers_influence import variance_inflation_factor


train_X_M = np.matrix(train_all_train[list(model.params.index)])

VIF_list = [variance_inflation_factor(train_X_M, i) for i in range(train_X_M.shape[1])]


### 重新训练模型 ##
model = smf.Logit(train_all_train['Defaulter'], train_all_train[list(model.params.index)]).fit()

### 
from sklearn.metrics import auc,roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score

## 用拟合好的模型预测训练集
## 首先将数据集的X和Y进行区分
train_all_train_X = train_all_train[list(model.params.index)]
train_all_train_Y = train_all_train['Defaulter']

train_all_test_X = train_all_test[list(model.params.index)]
train_all_test_Y = train_all_test['Defaulter']

y_train_proba = model.predict(train_all_train_X)

## 用拟合好的模型预测测试集
y_test_proba = model.predict(train_all_test_X)

### 计算训练集的AUC值
roc_auc_score(train_all_train_Y, y_train_proba)
### 计算测试集的AUC值
roc_auc_score(train_all_test_Y, y_test_proba)

import matplotlib.pyplot as plt
### 绘制roc曲线
fpr, tpr, thresholds = roc_curve(train_all_test_Y, y_test_proba, pos_label=1)
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




### 采用其他模型进行训练，评估效果

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier


x_col = list(set(train_all_train.columns).difference(set(['Defaulter'])))
train_all_train_X = train_all_train[x_col]
train_all_train_Y = train_all_train['Defaulter']

train_all_test_X = train_all_test[x_col]
train_all_test_Y = train_all_test['Defaulter']


## 建立不同的分类器模型 
model = GradientBoostingClassifier()

model.fit(train_all_train_X, train_all_train_Y)

## 用拟合好的模型预测训练集
y_train_proba = model.predict_proba(train_all_train_X)
y_train_label = model.predict(train_all_train_X)

## 用拟合好的模型预测测试集
y_test_proba = model.predict_proba(train_all_test_X)
y_test_label = model.predict(train_all_test_X)


print('训练集准确率：{:.2%}'.format(accuracy_score(train_all_train_Y, y_train_label)))
print('测试集准确率：{:.2%}'.format(accuracy_score(train_all_test_Y, y_test_label)))

print('训练集精度：{:.2%}'.format(precision_score(train_all_train_Y, y_train_label)))
print('测试集精度：{:.2%}'.format(precision_score(train_all_test_Y, y_test_label)))

print('训练集召回率：{:.2%}'.format(recall_score(train_all_train_Y, y_train_label)))
print('测试集召回率：{:.2%}'.format(recall_score(train_all_test_Y, y_test_label)))

print('训练集AUC：{:.2%}'.format(roc_auc_score(train_all_train_Y, y_train_proba[:,1])))
print('测试集AUC：{:.2%}'.format(roc_auc_score(train_all_test_Y, y_test_proba[:,1])))
  

### stacking 模型集成

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model.logistic import LogisticRegression


'''创建模型融合中的基模型'''
clfs = [AdaBoostClassifier(),
        RandomForestClassifier(n_estimators=50, n_jobs=-1, criterion='entropy'),
        LogisticRegression (C=0.01),
        ExtraTreesClassifier(n_estimators=50, n_jobs=-1, criterion='entropy'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

'''对数据集进行切分，切分为训练集和测试集'''

X_train, X_test,y_train,y_test = train_test_split(woe_train_X, train_Y, test_size=0.3)


dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_test.shape[0], len(clfs)))

'''5折stacking'''

n_folds = 5
c, r = y_train.shape
y_train = y_train.values.reshape(c,)
X_train = X_train.values

skf = list(StratifiedKFold(n_folds).split(X_train,y_train))

for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))
    
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        
        # print("Fold", i)
        X_train_kfold, y_train_kfold, X_test_kfold, y_test_kfold = X_train[train], y_train[train], X_train[test], y_train[test]
        clf.fit(X_train_kfold, y_train_kfold)
        y_submission = clf.predict_proba(X_test_kfold)[:, 1]
        dataset_blend_train[test, j] = y_submission
        
        dataset_blend_test_j[:, i] = clf.predict_proba(X_test)[:, 1]
    
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
    print("使用第" + str(j) + "个模型的：" + "Roc Auc Score: %f" % roc_auc_score(y_test, dataset_blend_test[:, j]))

# stacking 模型融合
clf = GradientBoostingClassifier(learning_rate=0.02, subsample=0.5, max_depth=6, n_estimators=30)
clf.fit(dataset_blend_train, y_train)
y_submission = clf.predict_proba(dataset_blend_test)[:, 1]


print("模型融合的结果：" + "Roc Auc Score: %f" % (roc_auc_score(y_test, y_submission)))
  









    

 


            
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    