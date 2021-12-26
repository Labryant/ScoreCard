
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression  as LR
plt.rcParams['font.sans-serif'] =['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] =False   #用来正常显示负号


# 按照Iv 值进行筛选
def select_iv(iv_df,ori_iv,thresold):
    """
    iv_df：已经分箱好的数据
    thresold:iv的比例
    """
    iv_df_1 = iv_df.copy()
    iv_df_1 = iv_df_1.loc[iv_df_1[ori_iv]>=thresold]
    return iv_df_1

# xgboost筛选变量 
def select_xgboost(df,target,imp_num=None):
    """
    df:数据集
    target:目标变量的字段名
    imp_num:筛选变量的个数
    
    return:
    xg_fea_imp:变量的特征重要性
    xg_select_col:筛选出的变量
    """
    x = df.drop([target],axis=1)
    y = df[target]
    xgmodel = XGBClassifier(random_state=0)
    xgmodel = xgmodel.fit(x,y,eval_metric='auc')
    xg_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':xgmodel.feature_importances_})
    xg_fea_imp = xg_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    xg_select_col = list(xg_fea_imp.col)
    return xg_fea_imp,xg_select_col


# 随机森林筛选变量 
def select_rf(df,target,imp_num=None):
    """
    df:数据集
    target:目标变量的字段名
    imp_num:筛选变量的个数
    
    return:
    rf_fea_imp:变量的特征重要性
    rf_select_col:筛选出的变量
    """
    x = df.drop([target],axis=1)
    y = df[target]
    rfmodel = RandomForestClassifier(random_state=0)
    rfmodel = rfmodel.fit(x,y)
    rf_fea_imp = pd.DataFrame({'col':list(x.columns),
                               'imp':rfmodel.feature_importances_})
    rf_fea_imp = rf_fea_imp.sort_values('imp',ascending=False).reset_index(drop=True).iloc[:imp_num,:]
    rf_select_col = list(rf_fea_imp.col)
    return rf_fea_imp,rf_select_col



# 相关性可视化
def plot_corr(df,col_list,threshold=None,plt_size=None,is_annot=True):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    plt_size:图纸尺寸
    is_annot:是否显示相关系数值
    
    return :相关性热力图
    """
    corr_df = df.loc[:,col_list].corr()
    plt.figure(figsize=plt_size)
    sns.heatmap(corr_df,annot=is_annot,cmap='rainbow',vmax=1,vmin=-1,mask=np.abs(corr_df)<=threshold)
    return plt.show()


# 相关性剔除
def forward_delete_corr(df,col_list,threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    
    return:相关性剔除后的变量
    """
    list_corr = col_list[:]
    for col in list_corr:
        corr = df.loc[:,list_corr].corr()[col]
        corr_index= [x for x in corr.index if x!=col]
        corr_values  = [x for x in corr.values if x!=1]
        for i,j in zip(corr_index,corr_values):
            if abs(j)>=threshold:
                list_corr.remove(i)
    return list_corr


# 相关性变量映射关系 
def corr_mapping(df,col_list,threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold: 相关性设定的阈值
    
    return:强相关性变量之间的映射关系表
    """
    corr_df = df.loc[:,col_list].corr()
    col_a = []
    col_b = []
    corr_value = []
    for col,i in zip(col_list[:-1],range(1,len(col_list),1)):
        high_corr_col=[]
        high_corr_value=[]
        corr_series = corr_df[col][i:]
        for i,j in zip(corr_series.index,corr_series.values):
            if abs(j)>=threshold:
                high_corr_col.append(i)
                high_corr_value.append(j)
        col_a.extend([col]*len(high_corr_col))
        col_b.extend(high_corr_col)
        corr_value.extend(high_corr_value)

    corr_map_df = pd.DataFrame({'col_A':col_a,
                                'col_B':col_b,
                                'corr':corr_value})
    return corr_map_df


# 显著性筛选,在筛选前需要做woe转换
def forward_delete_pvalue(x_train,y_train):
    """
    x_train -- x训练集
    y_train -- y训练集
    
    return :显著性筛选后的变量
    """
    col_list = list(x_train.columns)
    pvalues_col=[]
    for col in col_list:
        pvalues_col.append(col)
        x_train2 = sm.add_constant(x_train.loc[:,pvalues_col])
        sm_lr = sm.Logit(y_train,x_train2)
        sm_lr = sm_lr.fit()
        for i,j in zip(sm_lr.pvalues.index[1:],sm_lr.pvalues.values[1:]): 
            if j>=0.05:
                pvalues_col.remove(i)
    
    x_new_train = x_train.loc[:,pvalues_col]
    x_new_train2 = sm.add_constant(x_new_train)
    lr = sm.Logit(y_train,x_new_train2)
    lr = lr.fit()
    print(lr.summary2())
    return pvalues_col


# 逻辑回归系数符号筛选,在筛选前需要做woe转换
def forward_delete_coef(x_train,y_train):
    """
    x_train -- x训练集
    y_train -- y训练集
    
    return :
    coef_col回归系数符号筛选后的变量
    lr_coe：每个变量的系数值
    """
    col_list = list(x_train.columns)
    coef_col = []
    for col in col_list:
        coef_col.append(col)
        x_train2 = x_train.loc[:,coef_col]
        sk_lr = LogisticRegression(random_state=0).fit(x_train2,y_train)
        coef_df = pd.DataFrame({'col':coef_col,'coef':sk_lr.coef_[0]})
        if coef_df[coef_df.coef<0].shape[0]>0:
            coef_col.remove(col)
    
    x_new_train = x_train.loc[:,coef_col]
    lr = LogisticRegression(random_state=0).fit(x_new_train,y_train)
    lr_coe = pd.DataFrame({'col':coef_col,
                           'coef':lr.coef_[0]})
    return coef_col,lr_coe


# 基于L1 正则化筛选变量

def SelectData_l1(df,y):
  """
  基于L1正则化选取特征
  """
  x = df.drop(y,axis=1)
  y = df[y]
  lr = LR(penalty="l1",dual=False).fit(x,y)
  model = SelectFromModel(lr,prefit=True)
  var = []
  for i,j in enumerate(list(model.get_support())):
      if j == True:
          var.append(x.iloc[:,i].name)
  return var

"""逐步回归算法来筛选变量
这里包括向前或者向后来筛选变量
向前选择法考虑的调整的r方
"""
def SelectData_stepwise(df,target,intercept=True,normalize=False,criterion='bic',p_value_enter=0.05,
                        f_pvalue_enter =0.05,direction='backward',show_step=True,
                        criterion_enter = None,criterion_remove =None,max_iter =200,**kw):
    """逐步回归
    df: 数据集 response 为第一列
    response:str 回归相关的变量
    intercept:bool 模型是否存在截距
    criterion: str 默认是bic 逐步回归优化规则
    f_pavlue_enter: str 当选择criterion=’ssr‘时，模型加入或移除变量的f_pvalue阈值
    p_values_enter:float,默认是0.05 当选择criterion="both"时移除变量的pvalue的阈值
    direction：str  默认是backward
    show_step: 是否显示逐步回归的过程
    critertion_enter: 当选择的direction='both‘或’forward‘ 模型加入变量相应的critertion的阈值
    critertion_remove：当选择direction=backward时 模型移除变量的相应的criterion阈值
    max_iter:模型最大迭代次数  
    """
    critertion_list = ['bic','aic','ssr','rsquared','rsquared_adj']
    if criterion not in critertion_list:
        raise IOError('请输入正确的critertion，必须是以下内容之一:','\n',critertion_list)
    direction_list = ['backward','forward','both']
    
    if direction not in direction_list:
        raise IOError('请输入正确的direction,必须是以下内容之一:','\n',direction_list)
    
    # 默认p_enter 参数
    p_enter = {'bic':0.0,'aic':0.0,'ssr':0.05,'rsquared':0.05,'rsquared_adj':-0.05}
    
    if criterion_enter:
       p_enter[criterion] = criterion_enter
       
    # 默认的p_remove 参数
    p_remove = {'bic':0.01,'aic':0.01,'ssr':0.1,'rsquared':0.05,'rsquared_adj':-0.05}
    
    if criterion_remove:
        p_remove[criterion] = criterion_remove
    
    if normalize: #如果需要标准化数据
        intercept = False
        df_std = StandardScaler().fit_transform(df)
        df = pd.DataFrame(df_std,columns=df.columns,index=df.index)
        
    '''forward'''
    if direction == 'forward':
        remaining =  list(df.columns) #自变量集合
        remaining.remove(target)
        selected= [] #初始化选入模型的变量列表
        if intercept: #判断是否有截距
            formula  = "{}~{}+1".format(target,remaining[0])
        else:
            formula = "{}~{}-1".format(target,remaiing[0])
        result = smf.ols(formula,df).fit() #最小二乘法回归模型拟合
        current_score = eval('result.'+criterion)
        best_new_score = eval('result.'+criterion)
        if show_step:
            print('\nstepwise staring:\n')
        iter_times = 0
        #当变量未删除完，并且当前评分更新时进行循环
        while remaining and (current_score == best_new_score) and (iter_times < max_iter):
            scores_with_candidates = [] # 初始化变量以及评分列表
            for candidate in remaining: #在未删除的变量中每次选择一个变量进入模型，以此循环
                if intercept:
                    formula = "{}~{}+1".format(target,"+".join(selected+[candidate]))
                else:
                    formula = "{}~{}-1".format(target,"+".join(selected+[candidate]))
            result = smf.ols(formula,df).fit() #最小二乘法拟合
            fvalue = result.fvalue
            f_pvalue =result.f_pvalue
            score = eval('result.'+criterion)
            scores_with_candidates.append((score,candidate,fvalue,f_pvalue)) # 记录此次循环的变量、评分列表
        
            if criterion =='ssr': # 这几个指标取最小值进行优化
                scores_with_candidates.sort(reverse=True)
                best_new_score,best_candidate,best_new_fvalue,best_new_f_pvalue = scores_with_candidates.pop()
                if ((current_score - best_new_score) > p_enter[criterion]) and (
                        best_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
                    remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                    selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                    current_score = best_new_score  # 更新当前评分
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
                              (best_candidate, best_new_score, best_new_fvalue, best_new_f_pvalue))

                    elif (current_score - best_new_score) >= 0 and (
                            best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
                        remaining.remove(best_candidate)
                        selected.append(best_candidate)
                        current_score = best_new_score
                        if show_step:  # 是否显示逐步回归过程
                            print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                    elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                        selected.append(remaining[0])
                        remaining.remove(remaining[0])
                        if show_step:  # 是否显示逐步回归过程
                            print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
            elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                if (current_score - best_new_score) > p_enter[criterion]:  # 如果当前评分大于最新评分
                    remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                    selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                    current_score = best_new_score  # 更新当前评分
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif (current_score - best_new_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
            else:
                scores_with_candidates.sort()
                best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()
                if (best_new_score - current_score) > p_enter[criterion]:  # 当评分差大于p_enter
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif (best_new_score - current_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))

            if intercept:  # 是否有截距
                formula = "{} ~ {} + 1".format(target, ' + '.join(selected))
            else:
                formula = "{} ~ {} - 1".format(target, ' + '.join(selected))

            stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合

            if show_step:  # 是否显示逐步回归过程
                print('\nLinear regression model:', '\n  ', stepwise_model.model.formula)
                print('\n', stepwise_model.summary())

    ''' backward '''
    if direction == 'backward':
        remaining, selected = set(df.columns), set(df.columns)  # 自变量集合
        remaining.remove(target)
        selected.remove(target)  # 初始化选入模型的变量列表
        # 初始化当前评分,最优新评分
        if intercept:  # 是否有截距
            formula = "{} ~ {} + 1".format(target, ' + '.join(selected))
        else:
            formula = "{} ~ {} - 1".format(target, ' + '.join(selected))

        result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
        current_score = eval('result.' + criterion)
        worst_new_score = eval('result.' + criterion)

        if show_step:
            print('\nstepwise starting:\n')
        iter_times = 0
        # 当变量未剔除完，并且当前评分更新时进行循环
        while remaining and (current_score == worst_new_score) and (iter_times < max_iter):
            scores_with_eliminations = []  # 初始化变量以及其评分列表
            for elimination in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
                if intercept:  # 是否有截距
                    formula = "{} ~ {} + 1".format(target, ' + '.join(selected - set(elimination)))
                else:
                    formula = "{} ~ {} - 1".format(target, ' + '.join(selected - set(elimination)))

                result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
                fvalue = result.fvalue
                f_pvalue = result.f_pvalue
                score = eval('result.' + criterion)
                scores_with_eliminations.append((score, elimination, fvalue, f_pvalue))  # 记录此次循环的变量、评分列表

            if criterion == 'ssr':  # 这几个指标取最小值进行优化
                scores_with_eliminations.sort(reverse=False)  # 对评分列表进行降序排序
                worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()  # 提取最小分数及其对应变量
                if ((worst_new_score - current_score) < p_remove[criterion]) and (
                        worst_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
                    remaining.remove(worst_elimination)  # 从剩余未评分变量中剔除最新最优分对应的变量
                    selected.remove(worst_elimination)  # 从已选变量列表中剔除最新最优分对应的变量
                    current_score = worst_new_score  # 更新当前评分
                    if show_step:  # 是否显示逐步回归过程
                        print('Removing %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
                              (worst_elimination, worst_new_score, worst_new_fvalue, worst_new_f_pvalue))
            elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                scores_with_eliminations.sort(reverse=False)  # 对评分列表进行降序排序
                worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()  # 提取最小分数及其对应变量
                if (worst_new_score - current_score) < p_remove[criterion]:  # 如果评分变动不显著
                    remaining.remove(worst_elimination)  # 从剩余未评分变量中剔除最新最优分对应的变量
                    selected.remove(worst_elimination)  # 从已选变量列表中剔除最新最优分对应的变量
                    current_score = worst_new_score  # 更新当前评分
                    if show_step:  # 是否显示逐步回归过程
                        print('Removing %s, %s = %.3f' % (worst_elimination, criterion, worst_new_score))
            else:
                scores_with_eliminations.sort(reverse=True)
                worst_new_score, worst_elimination, worst_new_fvalue, worst_new_f_pvalue = scores_with_eliminations.pop()
                if (current_score - worst_new_score) < p_remove[criterion]:
                    remaining.remove(worst_elimination)
                    selected.remove(worst_elimination)
                    current_score = worst_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Removing %s, %s = %.3f' % (worst_elimination, criterion, worst_new_score))
            iter_times += 1

        if intercept:  # 是否有截距
            formula = "{} ~ {} + 1".format(target, ' + '.join(selected))
        else:
            formula = "{} ~ {} - 1".format(target, ' + '.join(selected))

        self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合

        if show_step:  # 是否显示逐步回归过程
            print('\nLinear regression model:', '\n  ',stepwise_model.model.formula)
            print('\n',stepwise_model.summary())

    ''' both '''
    if direction == 'both':
        remaining = list(df.columns)  # 自变量集合
        remaining.remove(target)
        selected = []  # 初始化选入模型的变量列表
        # 初始化当前评分,最优新评分
        if intercept:  # 是否有截距
            formula = "{} ~ {} + 1".format(target, remaining[0])
        else:
            formula = "{} ~ {} - 1".format(target, remaining[0])

        result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
        current_score = eval('result.' + criterion)
        best_new_score = eval('result.' + criterion)

        if show_step:
            print('\nstepwise starting:\n')
        # 当变量未剔除完，并且当前评分更新时进行循环
        iter_times = 0
        while remaining and (current_score == best_new_score) and (iter_times < max_iter):
            scores_with_candidates = []  # 初始化变量以及其评分列表
            for candidate in remaining:  # 在未剔除的变量中每次选择一个变量进入模型，如此循环
                if intercept:  # 是否有截距
                    formula = "{} ~ {} + 1".format(target, ' + '.join(selected + [candidate]))
                else:
                    formula = "{} ~ {} - 1".format(target, ' + '.join(selected + [candidate]))

                result = smf.ols(formula, df).fit()  # 最小二乘法回归模型拟合
                fvalue = result.fvalue
                f_pvalue = result.f_pvalue
                score = eval('result.' + criterion)
                scores_with_candidates.append((score, candidate, fvalue, f_pvalue))  # 记录此次循环的变量、评分列表

            if criterion == 'ssr':  # 这几个指标取最小值进行优化
                scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                if ((current_score - best_new_score) > p_enter[criterion]) and (
                        best_new_f_pvalue < f_pvalue_enter):  # 如果当前评分大于最新评分
                    remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                    selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                    current_score = best_new_score  # 更新当前评分
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, SSR = %.3f, Fstat = %.3f, FpValue = %.3e' %
                              (best_candidate, best_new_score, best_new_fvalue, best_new_f_pvalue))
                elif (current_score - best_new_score) >= 0 and (
                        best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif (best_new_f_pvalue < f_pvalue_enter) and iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
            elif criterion in ['bic', 'aic']:  # 这几个指标取最小值进行优化
                scores_with_candidates.sort(reverse=True)  # 对评分列表进行降序排序
                best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()  # 提取最小分数及其对应变量
                if (current_score - best_new_score) > p_enter[criterion]:  # 如果当前评分大于最新评分
                    remaining.remove(best_candidate)  # 从剩余未评分变量中剔除最新最优分对应的变量
                    selected.append(best_candidate)  # 将最新最优分对应的变量放入已选变量列表
                    current_score = best_new_score  # 更新当前评分
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif (current_score - best_new_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))
            else:
                scores_with_candidates.sort()
                best_new_score, best_candidate, best_new_fvalue, best_new_f_pvalue = scores_with_candidates.pop()
                if (best_new_score - current_score) > p_enter[criterion]:  # 当评分差大于p_enter
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif (best_new_score - current_score) >= 0 and iter_times == 0:  # 当评分差大于等于0，且为第一次迭代
                    remaining.remove(best_candidate)
                    selected.append(best_candidate)
                    current_score = best_new_score
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (best_candidate, criterion, best_new_score))
                elif iter_times == 0:  # 当评分差小于p_enter，且为第一次迭代
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    if show_step:  # 是否显示逐步回归过程
                        print('Adding %s, %s = %.3f' % (remaining[0], criterion, best_new_score))

            if intercept:  # 是否有截距
                formula = "{} ~ {} + 1".format(target, ' + '.join(selected))
            else:
                formula = "{} ~ {} - 1".format(target, ' + '.join(selected))

            result = smf.ols(formula, df).fit()  # 最优模型拟合
            if iter_times >= 1:  # 当第二次循环时判断变量的pvalue是否达标
                if result.pvalues.max() > p_value_enter:
                    var_removed = result.pvalues[result.pvalues == result.pvalues.max()].index[0]
                    p_value_removed = result.pvalues[result.pvalues == result.pvalues.max()].values[0]
                    selected.remove(result.pvalues[result.pvalues == result.pvalues.max()].index[0])
                    if show_step:  # 是否显示逐步回归过程
                        print('Removing %s, Pvalue = %.3f' % (var_removed, p_value_removed))
            iter_times += 1

        if intercept:  # 是否有截距
            formula = "{} ~ {} + 1".format(target, ' + '.join(selected))
        else:
            formula = "{} ~ {} - 1".format(target, ' + '.join(selected))

        self.stepwise_model = smf.ols(formula, df).fit()  # 最优模型拟合
        if show_step:  # 是否显示逐步回归过程
            print('\nLinear regression model:', '\n  ', stepwise_model.model.formula)
            print('\n', stepwise_model.summary())
            # 最终模型选择的变量
    if intercept:
        stepwise_feat_selected_ = list(stepwise_model.params.index[1:])
    else:
        stepwise_feat_selected_ = list(stepwise_model.params.index)
    return stepwise_feat_selected_




            
            
        
        
        
    
        
        
    
    
    
        
        
# def pick_variables(x,y,descover=True,method="rlr",threshold=0.25,sls=0.05):#默认阈值0.25
# #向后淘汰
#     if method =="bs"  and x.shape[1] > 1:
#         #提取X，y变量名
#         data = pd.concat([x, y], axis=1)#合并数据
#
#         var_list = x.columns
#         response = y.name
#         #首先对所有变量进行模型拟合
#         while True:
#             formula = "{} ~ {} + 1".format(response, ' + '.join(var_list))
#             mod = smf.logit(formula, data).fit()
#             print(mod.summary2())
#             p_list = mod.pvalues.sort_values()
#             if p_list[-1] > sls:
#                 #提取p_list中最后一个index
#                 var = p_list.index[-1]
#                 #var_list中删除
#                 var_list = var_list.drop(var)
#             else:
#                 break
#         x=x[var_list]
# #向前选择
#     if method =="fs":
#
#         data = pd.concat([x, y], axis=1)
#         response=y.name
#         remaining = set(x.columns)
#         selected = []
#         current_score, best_new_score = 0.0, 0.0
#         while remaining and current_score == best_new_score:
#             scores_with_candidates = []
#             for candidate in remaining:
#                 formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
#                 mod = smf.logit(formula, data).fit()
#                 score = mod.prsquared
#                 scores_with_candidates.append((score, candidate))
#             scores_with_candidates.sort(reverse=False)
#             best_new_score, best_candidate = scores_with_candidates.pop()
#             if current_score < best_new_score:
#                 remaining.remove(best_candidate)
#                 selected.append(best_candidate)
#                 current_score = best_new_score
#
#         print(len(selected))
#         x=x[selected]
#
# #rsquared_adj prsquared
#
#
#     if method =="fs_bs":
#         data = pd.concat([x, y], axis=1)
#         response=y.name
#
#         remaining = set(x.columns)
#         selected = []
#         current_score, best_new_score = 0.0, 0.0
#         while remaining and current_score == best_new_score:
#             scores_with_candidates = []
#             for candidate in remaining:
#                 formula = "{} ~ {} + 1".format(response, ' + '.join(selected + [candidate]))
#                 mod = smf.logit(formula, data).fit()
#                 score = mod.prsquared
#                 scores_with_candidates.append((score, candidate))
#             scores_with_candidates.sort(reverse=False)
#             best_new_score, best_candidate = scores_with_candidates.pop()
#             if current_score < best_new_score:
#                 print("===========================")
#                 remaining.remove(best_candidate)
#                 selected.append(best_candidate)
#                 current_score = best_new_score
#
#             formula2= "{} ~ {} + 1".format(response, ' + '.join(selected))
#             mod2 = smf.logit(formula2,data).fit()
#             p_list = mod2.pvalues.sort_values()
#             if p_list[-1] > sls:
#                 #提取p_list中最后一个index
#                 var = p_list.index[-1]
#                 #var_list中删除
#                 selected.remove(var)
#                 print(p_list[-1])
#                 formula3= "{} ~ {} + 1".format(response, ' + '.join(selected))
#
#                 mod3 = smf.logit(formula3, data).fit()
#                 best_new_score = mod3.prsquared
#                 current_score = best_new_score
#
#
#         print(len(selected))
#         x=x[selected]
#     '''
#     注意这里调用的是statsmodels.api里的逻辑回归。这个回归模型可以获取每个变量的显著性p值，p值越大越不显著，当我们发现多于一个变量不显著时，不能一次性剔除所有的不显著变量，因为里面可能存在我们还未发现的多变量的多重共线性，我们需要迭代的每次剔除最不显著的那个变量。
#     上面迭代的终止条件：
#     ①剔除了所有的不显著变量
#     ②剔除了某一个或某几个变量后，剩余的不显著变量变得显著了。（说明之前存在多重共线性）
#     '''
#     if method =="rfc":
#         RFC = RandomForestClassifier(n_estimators=200,max_depth=5,class_weight="balanced")
#         RFC_Model = RFC.fit(x,y)
#         features_rfc = x.columns
#         featureImportance = {features_rfc[i]:RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
#         featureImportanceSorted = sorted(featureImportance.items(),key=lambda x: x[1], reverse=True)
#         # we selecte the top 10 features
#         features_selection = [k[0] for k in featureImportanceSorted[:15]]
#
#         x = x[features_selection]
#         x['intercept'] = [1]*x.shape[0]
#
#         LR = sm.Logit(y, x).fit()
#         summary = LR.summary()
#         print(summary)
#         x=x.drop("intercept",axis=1)
#
#     return x





def binCreate(df,bins):
    colList = df.columns
    resDf = pd.DataFrame(columns=colList)
    m,n = df.shape
    referSer = pd.Series(range(m))
    referSer.name = 'rank'
    lableSer = pd.qcut(referSer, bins, labels=range(bins))
    lableSer.name = 'bin'
    lableDF = pd.concat([referSer,lableSer], axis=1)  #顺序与箱号合并
    for col in colList:
        rankSer = df[col].rank(method='min')
        rankSer.name = 'rank'
        rankDF = pd.concat([df[col],rankSer], axis=1)
        binsDF = pd.merge(rankDF, lableDF, on='rank', how='left')
        resDf[col] = binsDF['bin']
    return resDf

# 定义区间(类别)分布统计函数
def binDistStatistic(df,tag):
    colList = list(df.columns)  #转成列表
    colList.remove(tag)         #删除目标变量
    resDf = pd.DataFrame(columns=['colName','bin','binAllCnt','binPosCnt','binNegCnt','binPosRto','binNegRto'])
    for col in colList:
        allSer = df.groupby(col)[tag].count()         #计算样本数
        allSer = allSer[allSer>0]                     #剔除无效区间
        allSer.name = 'binAllCnt'                     #定义列名
        posSer = df.groupby(col)[tag].sum()           #计算正样本数
        posSer = posSer[allSer.index]                 #剔除无效区间
        posSer.name = 'binPosCnt'                     #定义列名
        tmpDf = pd.concat([allSer,posSer], axis=1)    #合并统计结果
        tmpDf = tmpDf.reset_index()                   #行索引转为一列
        tmpDf = tmpDf.rename(columns={col:'bin'})    #修改区间列列名
        tmpDf['colName'] = col                        #增加字段名称列
        tmpDf['binNegCnt'] = tmpDf['binAllCnt'] - tmpDf['binPosCnt']             #计算负样本数
        tmpDf['binPosRto'] = tmpDf['binPosCnt'] * 1.0000 / tmpDf['binAllCnt']    #计算正样本比例
        tmpDf['binNegRto'] = tmpDf['binNegCnt'] * 1.0000 / tmpDf['binAllCnt']    #计算负样本比例
        tmpDf = tmpDf.reindex(columns=['colName','bin','binAllCnt','binPosCnt','binNegCnt','binPosRto','binNegRto'])  #索引重排
        resDf = pd.concat([resDf,tmpDf])      #结果追加
    rows, cols = df.shape
    posCnt = df[tag].sum()
    resDf['allCnt'] = rows                              #总体样本数
    resDf['posCnt'] = posCnt                            #总体正样本数
    resDf['negCnt'] = rows - posCnt                     #总体负样本数
    resDf['posRto'] = posCnt * 1.0000 / rows            #总体正样本比例
    resDf['negRto'] = (rows - posCnt) * 1.0000 / rows   #总体负样本比例
    resDf['binPosCov'] = resDf['binPosCnt'] / resDf['posCnt']
    resDf['binNegCov'] = resDf['binNegCnt'] / resDf['negCnt']
    return resDf

# 定义区间(类别)属性统计函数
def binAttrStatistic(df,cont,disc,bins):
    m,n = df.shape
    referSer = pd.Series(range(m))
    referSer.name = 'rank'
    lableSer = pd.qcut(referSer, bins, labels=range(bins))
    lableSer.name = 'bin'
    lableDF = pd.concat([referSer,lableSer], axis=1)  #顺序与箱号合并
    resDf = pd.DataFrame(columns=['colName','bin','minVal','maxVal','binInterval'])
    for col in cont:
        rankSer = df[col].rank(method='min')
        rankSer.name = 'rank'
        rankDF = pd.concat([df[col],rankSer], axis=1)
        binsDF = pd.merge(rankDF, lableDF, on='rank', how='left')
        minSer = binsDF.groupby('bin')[col].min()
        minSer.name = 'minVal'
        maxSer = binsDF.groupby('bin')[col].max()
        maxSer.name = 'maxVal'
        tmpDf = pd.concat([minSer,maxSer], axis=1)
        tmpDf = tmpDf.reset_index()
        tmpDf['colName'] = col
        tmpDf['binInterval'] = tmpDf['minVal'].astype('str') + '-' + tmpDf['maxVal'].astype('str') 
        tmpDf = tmpDf.reindex(columns=['colName','bin','minVal','maxVal','binInterval'])
        tmpDf = tmpDf[tmpDf['binInterval']!='nan-nan']
        resDf =  pd.concat([resDf,tmpDf])
    for col in disc:
        binSer = pd.Series(df[col].unique())
        tmpDf = pd.concat([binSer,binSer], axis=1)
        tmpDf['colName'] = col
        tmpDf.rename(columns={0:'bin',1:'binInterval'}, inplace = True)
        tmpDf = tmpDf.reindex(columns=['colName','bin','minVal','maxVal','binInterval'])
        resDf = pd.concat([resDf,tmpDf])
    return resDf

# 定义结果合并函数
def binStatistic(df,cont,disc,tag,bins):
    binResDf = binCreate(df[cont], bins)  # 连续变量分箱
    binData = pd.concat([binResDf,df[disc],df[tag]], axis=1)  #合并离散变量与目标变量
    binDistStatResDf = binDistStatistic(binData,tag)  #对分箱后数据集进行分布统计
    binAttrStatResDf = binAttrStatistic(df,cont,disc,bins)  #区间(类别)大小统计
    binStatResDf = pd.merge(binDistStatResDf, binAttrStatResDf, left_on=['colName','bin'], right_on=['colName','bin'], how='left')
    resDf = binStatResDf.reindex(columns=['colName','bin','binInterval','minVal','maxVal','binAllCnt','binPosCnt','binNegCnt','binPosRto','binNegRto','allCnt','posCnt','negCnt','posRto','negRto','binPosCov','binNegCov'])
    return resDf

# 信息增益
import math
def entropyVal(prob):
    if (prob == 0 or prob == 1):
        entropy = 0
    else:
        entropy = -(prob * math.log(prob,2) + (1-prob) * math.log((1-prob),2))
    return entropy

def gain(df,cont,disc,tag,bins):
    binDf = binStatistic(df,cont,disc,tag,bins)
    binDf['binAllRto'] = binDf['binAllCnt'] / binDf['allCnt']   #计算各区间样本占比
    binDf['binEnty'] = binDf['binAllRto'] * binDf['binPosRto'].apply(entropyVal)    #计算各区间信息熵
    binDf['allEnty'] = binDf['posRto'].apply(entropyVal)        #计算总体信息熵
    tmpSer = binDf['allEnty'].groupby(binDf['colName']).mean() - binDf['binEnty'].groupby(binDf['colName']).sum()   #计算信息增益=总体信息熵-各区间信息熵加权和
    tmpSer.name = 'gain'
    resSer = tmpSer.sort_values(ascending=False)                #按信息增益大小降序重排
    return resSer

# 基于基尼系数

def giniVal(prob):
    gini = 1 - pow(prob,2) - pow(1-prob,2)
    return gini
    
def gini(df,cont,disc,tag,bins):
    binDf = binStatistic(df,cont,disc,tag,bins)
    binDf['binAllRto'] = binDf['binAllCnt'] / binDf['allCnt']   #计算各区间样本占比
    binDf['binGini'] = binDf['binAllRto'] * binDf['binPosRto'].apply(giniVal)    #计算各区间信息熵
    binDf['allGini'] = binDf['posRto'].apply(giniVal)        #计算总体信息熵
    tmpSer = binDf['allGini'].groupby(binDf['colName']).mean() - binDf['binGini'].groupby(binDf['colName']).sum()   #计算信息增益=总体信息熵-各区间信息熵加权和
    tmpSer.name = 'gini'
    resSer = tmpSer.sort_values(ascending=False)                #按信息增益大小降序重排
    return resSer

##区分度计算
def lift(df,cont,disc,tag,bins):
    binDf = binStatistic(df,cont,disc,tag,bins)
    binDf['binLift'] = binDf['binPosRto'] / binDf['posRto']             #区间提升度=区间正样本比例/总体正样本比例
    tmpSer = binDf['binLift'].groupby(binDf['colName']).max()           #变量区分度=max(区间提升度)
    tmpSer.name = 'lift'
    resSer = tmpSer.sort_values(ascending=False)        #按区分度大小降序重排
    return resSer


##信息值(IV)计算
def iv(df,cont,disc,tag,bins):
    binDf = binStatistic(df,cont,disc,tag,bins)
    binDf['binPosCovAdj'] = (binDf['binPosCnt'].replace(0,1)) / binDf['posCnt']     #调整后区间正样本覆盖率(避免值为0无法取对数)
    binDf['binNegCovAdj'] = (binDf['binNegCnt'].replace(0,1)) / binDf['negCnt']     #调整后区间负样本覆盖率(避免值为0无法取对数)
    binDf['woe'] = binDf['binPosCovAdj'].apply(lambda x:math.log(x,math.e)) - binDf['binNegCovAdj'].apply(lambda x:math.log(x,math.e))
    binDf['iv'] = binDf['woe'] * (binDf['binPosCovAdj'] - binDf['binNegCovAdj'])
    tmpSer = binDf.groupby('colName')['iv'].sum()
    tmpSer.name = 'iv'
    resSer = tmpSer.sort_values(ascending=False)
    return resSer
