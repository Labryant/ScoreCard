import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from random import shuffle

class LR_model:
    def __init__(self):
        self.model = 0

    def Build_LR(self, X, Y):
        lr = LogisticRegression(fit_intercept=True, penalty='l1')
        paras ={'C':[10**i for i in np.linspace(-3,2,20)]}
        models = GridSearchCV(param_grid=paras, estimator=lr)
        models.fit(X,Y)
        best_C = models.best_params_['C']
        lr_best = LogisticRegression(fit_intercept=True, penalty='l1',C=best_C)
        lr_best.fit(X, Y)
        self.model = lr_best
        return lr_best

    def Predict_LR(self, X_pred):
        Y_pred = self.model.predict_proba(X_pred)
        return Y_pred


folderOfData = '/Users/Code/Data Collections/bank default/'
all_data = pd.read_csv(folderOfData+'allData_3.csv', header=0,encoding='gbk')

train_data, test_data = train_test_split(all_data, test_size=10000)

features = [i for i in list(train_data.columns) if i.find('_WOE')>=0]
X_train, Y_train = train_data[features],train_data['target']
X_test, Y_test = test_data[features],test_data['target']
print(1/Y_train.mean()-1)


lr = LogisticRegression(fit_intercept=True, penalty='l1')
paras = {'C':np.arange(1,100,2)/10}
models = GridSearchCV(param_grid=paras, estimator=lr)
models.fit(X_train, Y_train)

best_model = LogisticRegression(fit_intercept=True, penalty='l1',C=models.best_params_['C'])
best_model.fit(X_train, Y_train)

simple_lr = LR_model()
simple_lr.Build_LR(X_train, Y_train)
simple_y_pred = simple_lr.Predict_LR(X_test)[:,1]
roc_auc_score(Y_test,simple_y_pred)  #0.75


####################
##### 欠采样方法 #####
####################
#简单欠采样
K = 2
good_train,bad_train = train_data[train_data['target'] == 0],train_data[train_data['target'] == 1]
good_sampled = good_train.sample(bad_train.shape[0]*K)
train_sampled = pd.concat([good_sampled,bad_train])

X_train, Y_train = train_sampled[features], train_sampled['target']
undersampling_lr = LR_model()
undersampling_lr.Build_LR(X_train, Y_train)
undersampling_y_pred = undersampling_lr.Predict_LR(X_test)[:,1]
roc_auc_score(Y_test,undersampling_y_pred)  #0.76


#多重欠采样
idx = list(good_train.Idx)
K = 10
shuffle(idx)
#使用K折交叉法，将数据集等分成K等分。又由于数据集的样本量未必是K的整倍数，因此第1~K-1份子集的大小是一致的，
# 第K份子集的大小不低于第1~K-1份子集的大小
sub_n = int(np.floor(len(idx)/K))
interval_starts = [i*sub_n for i in range(K)]
interval_ends = [(i+1)*sub_n-1 for i in range(K-1)] + [len(idx)-1]
ensemble_models=[]
for j in range(K):
    #拿出第j份子集与坏样本合并成为小训练集
    start_pot, end_pot = interval_starts[j],interval_ends[j]
    subset_idx = [idx[m] for m in range(start_pot,end_pot+1)]
    good_subset = good_train.loc[good_train['Idx'].isin(subset_idx)]
    train_subset = pd.concat([good_subset,bad_train])
    #训练LR
    subset_lr = LR_model().Build_LR(train_subset[features], train_subset['target'])
    ensemble_models.append(subset_lr)

ensemble_predicts = pd.DataFrame()
for i in range(K):
    sub_lr = ensemble_models[i]
    subset_y_pred = sub_lr.predict_proba(X_test)[:, 1]
    subset_y_pred_df = pd.DataFrame({"pred_"+str(i):subset_y_pred})
    ensemble_predicts = pd.concat([ensemble_predicts, subset_y_pred_df], axis=1)

ensemble_predicts['avg_pred'] = ensemble_predicts.apply(lambda x: np.mean(x), axis=1)
roc_auc_score(Y_test,ensemble_predicts['avg_pred'])   #0.77


########################
###### 加权逻辑回归 ######
########################
lr2 = LogisticRegression(fit_intercept=True, penalty='l1',class_weight='balanced')
paras = {'C':np.arange(1,100,2)/10}
models = GridSearchCV(param_grid=paras, estimator=lr2)
models.fit(X_train, Y_train)

best_weighted_lr = LogisticRegression(fit_intercept=True, penalty='l1',C=models.best_params_['C'],class_weight='balanced')
best_weighted_lr.fit(X_train, Y_train)
weighted_lr_pred = best_weighted_lr.predict_proba(X_test)[:, 1]
roc_auc_score(Y_test,weighted_lr_pred)     #0.76