
# 模型参数调优
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RandomizedLogisticRegression,RandomizedLasso,SGDClassifier,LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier

import xgboost as xgb
from bayes_opt import BayesianOptimization
from attrdict import AttrDict
import lightgbm as lgb

from sklearn.externals import joblib




def get_lr_formula(model,X):
    intercept = pd.DataFrame(model.intercept_)
    coef = model.coef_.T   #model.coef_ 模型的参数
    coef = pd.DataFrame(coef)
    formula = pd.concat([intercept,coef])
    index = ['Intercept']
    index = index + list(X.columns)
    formula.index = index
    formula.reset_index(inplace=True)
    formula.columns = ['参数','估计值']
    return formula
        


def pick_variables(x,y,descover=True,method="rlr",threshold=0.25,sls=0.05):#默认阈值0.25
    #挑选变量助手
    if method == "rlr":
        #随机逻辑回归选择与y线性关系的变量(稳定性选择1)。
        #在不同数据子集和特征子集上运行特征选择算法(rlr)，最终汇总选择结果
        rlr = RandomizedLogisticRegression(selection_threshold=threshold)
        rlr.fit(x,y)
        scoretable = pd.DataFrame(rlr.all_scores_,index = x.columns,columns = ['var_score'])
        columns_need = list(x.columns[rlr.get_support()])
        x = x[columns_need]
    #向后淘汰
    if method =="bs"  and x.shape[1] > 1: 
        #提取X，y变量名
        data = pd.concat([x, y], axis=1)#合并数据

        var_list = x.columns
        response = y.name
        #首先对所有变量进行模型拟合
        while True:
            formula = "{} ~ {} + 1".format(response, ' + '.join(var_list))
            mod = smf.logit(formula, data).fit()
            print(mod.summary2())
            p_list = mod.pvalues.sort_values()
            if p_list[-1] > sls:
                #提取p_list中最后一个index
                var = p_list.index[-1]
                #var_list中删除
                var_list = var_list.drop(var)           
            else:
                break
        x=x[var_list]
    #向前选择    
    if method =="fs":   
        data = pd.concat([x, y], axis=1)
        response=y.name
        remaining = set(x.columns)
        selected = []
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                mod = smf.logit(formula, data).fit()
                score = mod.prsquared
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort(reverse=False)
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score               
        print(len(selected))
        x=x[selected]

    #rsquared_adj prsquared
    if method =="fs_bs":  
        data = pd.concat([x, y], axis=1)
        response=y.name

        remaining = set(x.columns)
        selected = []
        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            for candidate in remaining:
                formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
                mod = smf.logit(formula, data).fit()
                score = mod.prsquared
                scores_with_candidates.append((score, candidate))
            scores_with_candidates.sort(reverse=False)
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score:
                print("===========================")
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
            
            formula2= "{} ~ {} + 1".format(response, ' + '.join(selected))
            mod2 = smf.logit(formula2,data).fit()
            p_list = mod2.pvalues.sort_values()
            if p_list[-1] > sls:
                #提取p_list中最后一个index
                var = p_list.index[-1]
                #var_list中删除
                selected.remove(var)
                print(p_list[-1])
                formula3= "{} ~ {} + 1".format(response, ' + '.join(selected))
                
                mod3 = smf.logit(formula3, data).fit()
                best_new_score = mod3.prsquared
                current_score = best_new_score 
        print(len(selected))
        x=x[selected]

    '''
    注意这里调用的是statsmodels.api里的逻辑回归。这个回归模型可以获取每个变量的显著性p值，p值越大越不显著，当我们发现多于一个变量不显著时，
    不能一次性剔除所有的不显著变量，因为里面可能存在我们还未发现的多变量的多重共线性，我们需要迭代的每次剔除最不显著的那个变量。 
    上面迭代的终止条件： 
    ①剔除了所有的不显著变量 
    ②剔除了某一个或某几个变量后，剩余的不显著变量变得显著了。（说明之前存在多重共线性）
    '''
    if method =="rfc":   
        RFC = RandomForestClassifier(n_estimators=200,max_depth=5,class_weight="balanced")
        RFC_Model = RFC.fit(x,y)
        features_rfc = x.columns
        featureImportance = {features_rfc[i]:RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
        featureImportanceSorted = sorted(featureImportance.items(),key=lambda x: x[1], reverse=True)
        features_selection = [k[0] for k in featureImportanceSorted[:15]] 
        x = x[features_selection]
        x['intercept'] = [1]*x.shape[0]
        LR = sm.Logit(y, x).fit()
        summary = LR.summary()
        print(summary)
        x=x.drop("intercept",axis=1)
    return x



def model_optimizing(x,y,model="LR"):
    if model == "LR":
        pipline = Pipeline([('lr',LogisticRegression(class_weight="balanced"))])
        parameters = {
          #C正则化的系数
          'lr__penalty': ('l1','l2'),'lr__C': (0.01,0.1,10,1),'lr__max_iter':(80,150,100)}
    elif model=="sgd":
        pipline = Pipeline([
                ('sgd',SGDClassifier(loss='log'))#LR
                #('sgd',SGDClassifier(loss='hinge'))#SVM
                #('svm',SVC()) 
                ])
        parameters = {
              #随机梯度下降分类器。alpha正则化的系数,n_iter在训练集训练的次数，learning_rate为什么是alpha的倒数
              'sgd__alpha':(0.00001,0.000001,0.0001),'sgd__penalty':('l1','l2','elasticnet'),'sgd__n_iter':(10,50,5),  
              #核函数，将数据映射到高维空间中，寻找可区分数据的高维空间的超平面
              #'svm__C':(2.5,1),'svm__kernel':('linear','poly','rbf'),
          }
   
    grid_search = GridSearchCV(pipline,parameters,n_jobs=6,scoring='recall',cv=5)
    grid_search.fit(x, y) 
    print('Best score: %0.3f' % grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name])) 
    print("Grid scores on development set:")
    print()
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
    return best_parameters


def xgb_model_fit(params, dtrain, max_round=300, cv_folds=5, n_stop_round=50):
    """对一组参数进行交叉验证，并返回最优迭代次数和最优的结果。
    Args:
        params: dict, xgb 模型参数。
        见 xgb_grid_search_cv 函数

    Returns: 
        n_round: 最优迭代次数
        mean_auc: 最优的结果
    """
    cv_result = xgb.cv(params, dtrain, max_round, nfold=cv_folds,
        metrics='auc', early_stopping_rounds=n_stop_round, show_stdv=False)
    n_round = cv_result.shape[0]  # 最优模型，最优迭代次数
    mean_auc = cv_result['test-auc-mean'].values[-1]  # 最好的  AUC
    return n_round, mean_auc


def xgb_grid_search_cv( key, search_params,params=None,, dtrain=None, max_round=300, cv_folds=5,n_stop_round=50, return_best_model=True, verbose=True):
    """自定义 grid_search_cv for xgboost 函数。
    Args: 
        params: dict, xgb 模型参数。
        key: 待搜寻的参数。
        search_params：list, 待搜寻的参数list。
        dtrain： 训练数据
        max_round: 最多迭代次数
        cv_folds: 交叉验证的折数
        early_stopping_rounds: 迭代多少次没有提高则停止。
        return_best_model: if True, 在整个训练集上使用最优的参数训练模型。
        verbose：if True, 打印训练过程。

    Returns:
        cv_results: dict，所有参数组交叉验证的结果。
            - mean_aucs: 每组参数对应的结果。
            - n_rounds: 每组参数最优迭代轮数。
            - list_params: 搜寻的每一组参数。
            - best_mean_auc: 最优的结果。
            - best_round: 最优迭代轮数。
            - best_params: 最优的一组参数。
        best_model: XGBoostClassifer() 
    """  
    if params==None:
        params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'learning_rate': 0.1,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'eta': 0.1,
          'max_depth': 5,
          'min_child_weight': 1,
          'gamma': 0.0,
          'silent': 1,
          'seed': 0,
          'eval_metric': 'auc',
          'njob':8
          }        
   
    mean_aucs = list()
    n_rounds = list()
    list_params = list()
    print('Searching parameters: %s %s' % (key, str(search_params)))
    tic = time.time()
    for search_param in search_params:
        params[key] = search_param
        list_params.append(params.copy())
        n_round, mean_auc = xhb_model_fit(params, dtrain, max_round, cv_folds, n_stop_round)
        if verbose:
            print('%s=%s: n_round=%d, mean_auc=%g. Time cost %gs' % (key, str(search_param), n_round, mean_auc, time.time() - tic))
        mean_aucs.append(mean_auc)
        n_rounds.append(n_round)
    best_mean_auc = max(mean_aucs)
    best_index = mean_aucs.index(best_mean_auc)  # 最优的一组
    best_round = n_rounds[best_index]
    best_params = list_params[best_index]
    cv_result = {'mean_aucs': mean_aucs, 'n_rounds': n_rounds, 'list_params': list_params, 
                'best_mean_auc': best_mean_auc, 'best_round': best_round, 'best_params': best_params}
    if return_best_model:       
        best_model = xgb.train(best_params, dtrain, num_boost_round=best_round)
    else:
        best_model = None
    if verbose:
        print('best_mean_auc = %g' % best_mean_auc)
        print('best_round = %d' % best_round)
        print('best_params = %s' % str(best_params))
    return cv_result, best_model


def xgb_eval(max_depth,min_child_weight,reg_alpha,gamma,subsample,colsample_bytree):
    """对一组参数进行交叉验证，并返回最优迭代次数和最优的结果。
    Args:
        params: dict, xgb 模型参数。
        见 xgb_grid_search_cv 函数

    Returns: 
        n_round: 最优迭代次数
        mean_auc: 最优的结果
    """
    params = {'booster': 'gbtree',
          'objective': 'binary:logistic',
          'learning_rate': 0.1,
          'subsample': subsample,
          'colsample_bytree': colsample_bytree,
          'eta': 0.1,
          'reg_alpha':reg_alpha,
          'max_depth':int(max_depth),
          'min_child_weight': min_child_weight,
          'gamma':gamma,
          'silent': 1,
          'seed': 0,
          'eval_metric': 'auc',
          'njob':8
          }
    cv_result = xgb.cv(params,dtrain, 300, nfold=5,metrics='auc',early_stopping_rounds=50,show_stdv=False)
    n_round = cv_result.shape[0]  # 最优模型，最优迭代次数
    mean_auc = cv_result['test-auc-mean'].values[-1]  # 最好的  AUC
    return  mean_auc
     
     
# xgboost 调优
    
class XGBoost:
    def __init__(self, **params):
        super().__init__()
        print('initializing XGBoost...')
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
        self.evaluation_function = None
        if 'number_boosting_rounds'not in  self.params.keys():
            self.params['number_boosting_rounds']=1000
        if 'early_stopping_rounds'not in  self.params.keys():
            self.params['early_stopping_rounds']=50
        if 'verbose'not in  self.params.keys():
            self.params['verbose']=10
        if 'metric'not in  self.params.keys():
            self.params['metric']='auc'
        if 'metric'not in  self.params.keys():
            self.params['metric']='auc'

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self,
           train_x,train_y,
            valid_x,valid_y,
            **kwargs):
        print('xgboost, train data shape        {}'.format(X.shape))
        print('xgboost, validation data shape   {}'.format(X_valid.shape))
        print('xgboost, train labels shape      {}'.format(y.shape))
        print('xgboost, validation labels shape {}'.format(y_valid.shape))

        self.estimator=xgb.XGBClassifier(**self.model_config)
        self.estimator.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                                            eval_metric= self.model_config.metric, 
                                            verbose= self.model_config.verbose, 
                                            early_stopping_rounds= self.training_config.early_stopping_rounds,
                                            **kwargs)
    
    def cv_(self,
            X, y,
            X_valid, y_valid,
            feature_names=None,
            feature_types=None,
            **kwargs):
        train = xgb.DMatrix(X,
                            label=y,
                            feature_names=feature_names,
                            feature_types=feature_types)
        valid = xgb.DMatrix(X_valid,
                            label=y_valid,
                            feature_names=feature_names,
                            feature_types=feature_types)

        evaluation_results = {}
        self.cv_result = xgb.cv(params=self.model_config, dtrain=train, nfold=5,
        metrics=metrics, early_stopping_rounds=self.training_config.early_stopping_rounds, show_stdv=False)
        n_round = self.cv_result.shape[0]  # 最优模型，最优迭代次数
        mean_auc = self.cv_result['test-auc-mean'].values[-1] 
        print("mean_auc:{}".format(mean_auc))
        return mean_auc
    
    def load(self, filepath):
        self.estimator = xgb.Booster(params=self.model_config)
        self.estimator.load_model(filepath)
        return self

    def persist(self, filepath):
        self.estimator.save_model(filepath)

def xgb_bayesopter(x,y,max_depth=(3, 8.99),min_data_in_leaf= (20, 100),
                                             num_leaves= (20, 100),
                                             min_child_weight= (1, 100),
                                             reg_alpha=(0.00001, 0.2),
                                             reg_lambda= (0.00001, 0.2),
                                             min_split_gain=(0.00001, 0.1),
                                             subsample=(0.2, 1.0),
                                             colsample_bytree=(0.2, 1.0)):

    def xgb_eval(max_depth,min_data_in_leaf,num_leaves,min_child_weight
                 ,reg_alpha,reg_lambda,min_split_gain,subsample,colsample_bytree):
        
                 model=XGBoost(nthread=4,
                    n_estimators=1000,
                    learning_rate=0.02,
                    num_leaves=int(num_leaves),
                    colsample_bytree=max(min(colsample_bytree, 1), 0),
                    subsample= max(min(subsample, 1), 0),
                    max_depth=int(max_depth),
                    reg_alpha=max(reg_alpha, 0),
                    reg_lambda=max(reg_lambda, 0),
                    min_split_gain=min_split_gain,
                    min_child_weight=min_child_weight,
                    silent=-1,
                    verbose=10,
                    early_stopping_rounds= 30,number_boosting_rounds=1000,metric="auc")
                 
                 return model.cv_(x,y)
     
     
    clf_bo = BayesianOptimization(xgb_eval, {'max_depth': max_depth,
                                             'min_data_in_leaf': max_depth,
                                             'num_leaves': num_leaves,
                                             'min_child_weight': min_child_weight,
                                             'reg_alpha': reg_alpha,
                                             'reg_lambda': reg_lambda,
                                             'min_split_gain':min_split_gain,
                                             'subsample': subsample,
                                             'colsample_bytree' :colsample_bytree})
     
    clf_bo.maximize(init_points=4, n_iter=20)
    return clf_bo.res['max']


# LightGBM 调优
class LightGBM():
    '''
    最少需要输入四个参数：number_boosting_rounds=1000,early_stopping_rounds=30,verbose=1,metric="auc"
    '''
    
    def __init__(self, name=None, **params):
        print('initializing LightGBM...')
        self.params = params
        self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
        self.evaluation_function = None
        self.name="gbm"
        if 'number_boosting_rounds'not in  self.params.keys():
            self.params['number_boosting_rounds']=1000
        if 'early_stopping_rounds'not in  self.params.keys():
            self.params['early_stopping_rounds']=50
        if 'verbose'not in  self.params.keys():
            self.params['verbose']=10
        if 'metric'not in  self.params.keys():
            self.params['metric']='auc'
        if 'metric'not in  self.params.keys():
            self.params['metric']='auc'

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self,
           train_x,train_y,
            valid_x,valid_y,
            **kwargs):
        evaluation_results = {}

        print('LightGBM, train data shape        {}'.format(X.shape))
        print('LightGBM, validation data shape   {}'.format(X_valid.shape))
        print('LightGBM, train labels shape      {}'.format(y.shape))
        print('LightGBM, validation labels shape {}'.format(y_valid.shape))


        self.estimator = lgb.LGBMClassifier(**self.model_config,
                                     n_estimators=self.training_config.number_boosting_rounds,)

        self.estimator.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                                            eval_metric= self.model_config.metric, 
                                            verbose= self.model_config.verbose, 
                                            early_stopping_rounds= self.training_config.early_stopping_rounds,
                                            **kwargs)
    
    def cv_(self,X,y,
            feature_names='auto',
            categorical_features='auto',

            **kwargs,):
        
        data_train = lgb.Dataset(data=X,
                                 label=y,
                                 feature_name=feature_names,
                                 categorical_feature=categorical_features,
                                 **kwargs)
        self.cv_result=lgb.cv(self.model_config,data_train,
                              num_boost_round=self.training_config.number_boosting_rounds,
                              early_stopping_rounds=self.training_config.early_stopping_rounds,
                              verbose_eval=self.model_config.verbose,
                              )
        cv_mean=self.cv_result[self.model_config.metric  +  "-mean"][-1]
        return cv_mean
    
    def kfold(self,train_df, test_df, stratified = False,target="TARGET",**kwargs,):

        # Divide in training/validation and test data
        print("Starting kfold LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
        # Cross validation model
        if stratified:
            folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
        else:
            folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
        # Create arrays and dataframes to store results
        oof_preds = np.zeros(train_df.shape[0])
        sub_preds = np.zeros(test_df.shape[0])
        feature_importance_df = pd.DataFrame()
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df.drop([target],axis=1), train_df[target])):
            train_x, train_y = train_df.drop([target],axis=1).iloc[train_idx], train_df[target].iloc[train_idx]
            valid_x, valid_y = train_df.drop([target],axis=1).iloc[valid_idx], train_df[target].iloc[valid_idx]
    
            # LightGBM parameters found by Bayesian optimization
            clf = lgb.LGBMClassifier(**self.model_config,
                                     n_estimators=self.training_config.number_boosting_rounds,)

            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                                            eval_metric= self.model_config.metric, 
                                            verbose= self.model_config.verbose, 
                                            early_stopping_rounds= self.training_config.early_stopping_rounds)

            oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = feats
            fold_importance_df["importance"] = clf.feature_importances_
            fold_importance_df["fold"] = n_fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
            
            del clf, train_x, train_y, valid_x, valid_y
            gc.collect()
        print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
        # Write submission file and plot feature importance
        test_df['TARGET'] = sub_preds

        self.display_importances(feature_importance_df)
        self.feature_importance_df=feature_importance_df

        return feature_importance_df
    
    # Display/plot feature importance
    def display_importances(self,feature_importance_df_):
        cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
        plt.figure(figsize=(8, 10))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()

    def load(self, filepath):
        self.estimator = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.estimator, filepath)

    def _check_target_shape_and_type(self, target, name):
        if not any([isinstance(target, obj_type) for obj_type in [pd.Series, np.ndarray, list]]):
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(target)))
        try:
            assert len(target.shape) == 1, '"{}" must be 1-D. It is {}-D instead.'.format(name,
                                                                                          len(target.shape))
        except AttributeError:
            print('Cannot determine shape of the {}. '
                  'Type must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead'.format(name,
                                                                                                     type(target)))

    def _format_target(self, target):

        if isinstance(target, pd.Series):
            return target.values
        elif isinstance(target, np.ndarray):
            return target
        elif isinstance(target, list):
            return np.array(target)
        else:
            raise TypeError(
                '"target" must be "numpy.ndarray" or "Pandas.Series" or "list", got {} instead.'.format(type(target)))

def lgbm_bayesopter(x,y,max_depth=(3, 8.99),min_data_in_leaf= (20, 100),
                                             num_leaves= (20, 100),
                                             min_child_weight= (1, 100),
                                             reg_alpha=(0.00001, 0.2),
                                             reg_lambda= (0.00001, 0.2),
                                             min_split_gain=(0.00001, 0.1),
                                             subsample=(0.2, 1.0),
                                             colsample_bytree=(0.2, 1.0)):

    def lgbm_eval(max_depth,min_data_in_leaf,num_leaves,min_child_weight
                 ,reg_alpha,reg_lambda,min_split_gain,subsample,colsample_bytree):
        
                 model=LightGBM(nthread=4,
                    n_estimators=10000,
                    learning_rate=0.02,
                    num_leaves=int(num_leaves),
                    colsample_bytree=max(min(colsample_bytree, 1), 0),
                    subsample= max(min(subsample, 1), 0),
                    max_depth=int(max_depth),
                    reg_alpha=max(reg_alpha, 0),
                    reg_lambda=max(reg_lambda, 0),
                    min_split_gain=min_split_gain,
                    min_child_weight=min_child_weight,
                    silent=-1,
                    verbose=10,
                    early_stopping_rounds= 30,number_boosting_rounds=1000,metric="auc")
                 
                 return model.cv_(x,y)
     
     
    clf_bo = BayesianOptimization(lgbm_eval, {'max_depth': max_depth,
                                             'min_data_in_leaf': max_depth,
                                             'num_leaves': num_leaves,
                                             'min_child_weight': min_child_weight,
                                             'reg_alpha': reg_alpha,
                                             'reg_lambda': reg_lambda,
                                             'min_split_gain':min_split_gain,
                                             'subsample': subsample,
                                             'colsample_bytree' :colsample_bytree})
     
    clf_bo.maximize(init_points=4, n_iter=20)
    return clf_bo
    