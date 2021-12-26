
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
plt.rcParams['font.sans-serif'] =['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] =False   #用来正常显示负号
# 变量分箱

# 类别性变量的分箱 
def binning_cate(df,col_list,target):
    """
    df:数据集
    col_list:变量list集合
    target:目标变量的字段名
    
    return: 
    bin_df :list形式，里面存储每个变量的分箱结果
    iv_value:list形式，里面存储每个变量的IV值
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total-bad
    all_odds = good*1.0/bad
    bin_df =[]
    iv_value=[]
    for col in col_list:
        d1 = df.groupby([col],as_index=True)
        d2 = pd.DataFrame()
        d2['min_bin'] = d1[col].min()
        d2['max_bin'] = d1[col].max()
        d2['total'] = d1[target].count()
        d2['totalrate'] = d2['total']/total
        d2['bad'] = d1[target].sum()
        d2['badrate'] = d2['bad']/d2['total']
        d2['good'] = d2['total'] - d2['bad']
        d2['goodrate'] = d2['good']/d2['total']
        d2['badattr'] = d2['bad']/bad
        d2['goodattr'] = (d2['total']-d2['bad'])/good
        d2['odds'] = d2['good']/d2['bad']
        GB_list=[]
        for i in d2.odds:
            if i>=all_odds:
                GB_index = str(round((i/all_odds)*100,0))+str('G')
            else:
                GB_index = str(round((all_odds/i)*100,0))+str('B')
            GB_list.append(GB_index)
        d2['GB_index'] = GB_list
        d2['woe'] = np.log(d2['badattr']/d2['goodattr'])
        d2['bin_iv'] = (d2['badattr']-d2['goodattr'])*d2['woe']
        d2['IV'] = d2['bin_iv'].sum()
        df2["badcumsum"] = df2['bad'].cumsum()
        df2['goodcumsum'] = df2['good'].cumsum()
        df2['ks'] = max(df2['badcumsum']/sum(df2['badcumsum']) - df2['goodcumsum']/sum(df2['goodcumsum']))
        iv = d2['bin_iv'].sum().round(3)
        print('变量名:{}'.format(col))
        print('IV:{}'.format(iv))
        print("KS:{}".format(df2['ks'].values.unique))
        print('\t')
        bin_df.append(d2)
        iv_value.append(iv)
    return bin_df,iv_value


# 类别性变量iv的明细表
def iv_cate(df,col_list,target):
    """
    df:数据集
    col_list:变量list集合
    target:目标变量的字段名
    
    return:变量的iv明细表
    """
    bin_df,iv_value = binning_cate(df,col_list,target)
    iv_df = pd.DataFrame({'col':col_list,
                          'iv':iv_value})
    iv_df = iv_df.sort_values('iv',ascending=False)
    return iv_df


# 数值型变量的分箱 

# 先用卡方分箱输出变量的分割点
def split_data(df,col,split_num):
    """
    df: 原始数据集
    col:需要分箱的变量
    split_num:分割点的数量
    """
    df2 = df.copy()
    count = df2.shape[0] # 总样本数
    n = math.floor(count/split_num) # 按照分割点数目等分后每组的样本数
    split_index = [i*n for i in range(1,split_num)] # 分割点的索引
    values = sorted(list(df2[col])) # 对变量的值从小到大进行排序
    split_value = [values[i] for i in split_index] # 分割点对应的value
    split_value = sorted(list(set(split_value))) # 分割点的value去重排序
    return split_value

def assign_group(x,split_bin):
    """
    x:变量的value
    split_bin:split_data得出的分割点list
    """
    n = len(split_bin)
    if x<=min(split_bin):   
        return min(split_bin) # 如果x小于分割点的最小值，则x映射为分割点的最小值
    elif x>max(split_bin): # 如果x大于分割点的最大值，则x映射为分割点的最大值
        return 10e10
    else:
        for i in range(n-1):
            if split_bin[i]<x<=split_bin[i+1]:# 如果x在两个分割点之间，则x映射为分割点较大的值
                return split_bin[i+1]

def bin_bad_rate(df,col,target,grantRateIndicator=0):
    """
    df:原始数据集
    col:原始变量/变量映射后的字段
    target:目标变量的字段
    grantRateIndicator:是否输出总体的违约率
    """
    total = df.groupby([col])[target].count()
    bad = df.groupby([col])[target].sum()
    total_df = pd.DataFrame({'total':total})
    bad_df = pd.DataFrame({'bad':bad})
    regroup = pd.merge(total_df,bad_df,left_index=True,right_index=True,how='left')
    regroup = regroup.reset_index()
    regroup['bad_rate'] = regroup['bad']/regroup['total']  # 计算根据col分组后每组的违约率
    dict_bad = dict(zip(regroup[col],regroup['bad_rate'])) # 转为字典形式
    if grantRateIndicator==0:
        return (dict_bad,regroup)
    total_all= df.shape[0]
    bad_all = df[target].sum()
    all_bad_rate = bad_all/total_all # 计算总体的违约率
    return (dict_bad,regroup,all_bad_rate)

def cal_chi2(df,all_bad_rate):
    """
    df:bin_bad_rate得出的regroup
    all_bad_rate:bin_bad_rate得出的总体违约率
    """
    df2 = df.copy()
    df2['expected'] = df2['total']*all_bad_rate # 计算每组的坏用户期望数量
    combined = zip(df2['expected'],df2['bad']) # 遍历每组的坏用户期望数量和实际数量
    chi = [(i[0]-i[1])**2/i[0] for i in combined] # 计算每组的卡方值
    chi2 = sum(chi) # 计算总的卡方值
    return chi2

def assign_bin(x,cutoffpoints):
    """
    x:变量的value
    cutoffpoints:分箱的切割点
    """
    bin_num = len(cutoffpoints)+1 # 箱体个数
    if x<=cutoffpoints[0]:  # 如果x小于最小的cutoff点，则映射为Bin 0
        return 'Bin 0'
    elif x>cutoffpoints[-1]: # 如果x大于最大的cutoff点，则映射为Bin(bin_num-1)
        return 'Bin {}'.format(bin_num-1)
    else:
        for i in range(0,bin_num-1):
            if cutoffpoints[i]<x<=cutoffpoints[i+1]: # 如果x在两个cutoff点之间，则x映射为Bin(i+1)
                return 'Bin {}'.format(i+1)

def ChiMerge(df,col,target,max_bin=5,min_binpct=0):
    col_unique = sorted(list(set(df[col]))) # 变量的唯一值并排序
    n = len(col_unique) # 变量唯一值得个数
    df2 = df.copy()
    if n>100:  # 如果变量的唯一值数目超过100，则将通过split_data和assign_group将x映射为split对应的value
        split_col = split_data(df2,col,100)  # 通过这个目的将变量的唯一值数目人为设定为100
        df2['col_map'] = df2[col].map(lambda x:assign_group(x,split_col))
    else:
        df2['col_map'] = df2[col]  # 变量的唯一值数目没有超过100，则不用做映射
    # 生成dict_bad,regroup,all_bad_rate的元组
    (dict_bad,regroup,all_bad_rate) = bin_bad_rate(df2,'col_map',target,grantRateIndicator=1)
    col_map_unique = sorted(list(set(df2['col_map'])))  # 对变量映射后的value进行去重排序
    group_interval = [[i] for i in col_map_unique]  # 对col_map_unique中每个值创建list并存储在group_interval中
    
    while (len(group_interval)>max_bin): # 当group_interval的长度大于max_bin时，执行while循环
        chi_list=[]
        for i in range(len(group_interval)-1):
            temp_group = group_interval[i]+group_interval[i+1] # temp_group 为生成的区间,list形式，例如[1,3]
            chi_df = regroup[regroup['col_map'].isin(temp_group)]
            chi_value = cal_chi2(chi_df,all_bad_rate) # 计算每一对相邻区间的卡方值
            chi_list.append(chi_value)
        best_combined = chi_list.index(min(chi_list)) # 最小的卡方值的索引
        # 将卡方值最小的一对区间进行合并
        group_interval[best_combined] = group_interval[best_combined]+group_interval[best_combined+1]
        # 删除合并前的右区间
        group_interval.remove(group_interval[best_combined+1])
        # 对合并后每个区间进行排序
    group_interval = [sorted(i) for i in group_interval]
    # cutoff点为每个区间的最大值
    cutoffpoints = [max(i) for i in group_interval[:-1]]
    
    # 检查是否有箱只有好样本或者只有坏样本
    df2['col_map_bin'] = df2['col_map'].apply(lambda x:assign_bin(x,cutoffpoints)) # 将col_map映射为对应的区间Bin
    # 计算每个区间的违约率
    (dict_bad,regroup) = bin_bad_rate(df2,'col_map_bin',target)
    # 计算最小和最大的违约率
    [min_bad_rate,max_bad_rate] = [min(dict_bad.values()),max(dict_bad.values())]
    # 当最小的违约率等于0，说明区间内只有好样本，当最大的违约率等于1，说明区间内只有坏样本
    while min_bad_rate==0 or max_bad_rate==1:
        bad01_index = regroup[regroup['bad_rate'].isin([0,1])].col_map_bin.tolist()# 违约率为1或0的区间
        bad01_bin = bad01_index[0]
        if bad01_bin==max(regroup.col_map_bin):
            cutoffpoints = cutoffpoints[:-1] # 当bad01_bin是最大的区间时，删除最大的cutoff点
        elif bad01_bin==min(regroup.col_map_bin):
            cutoffpoints = cutoffpoints[1:] # 当bad01_bin是最小的区间时，删除最小的cutoff点
        else:
            bad01_bin_index = list(regroup.col_map_bin).index(bad01_bin) # 找出bad01_bin的索引
            prev_bin = list(regroup.col_map_bin)[bad01_bin_index-1] # bad01_bin前一个区间
            df3 = df2[df2.col_map_bin.isin([prev_bin,bad01_bin])] 
            (dict_bad,regroup1) = bin_bad_rate(df3,'col_map_bin',target)
            chi1 = cal_chi2(regroup1,all_bad_rate)  # 计算前一个区间和bad01_bin的卡方值
            later_bin = list(regroup.col_map_bin)[bad01_bin_index+1] # bin01_bin的后一个区间
            df4 = df2[df2.col_map_bin.isin([later_bin,bad01_bin])] 
            (dict_bad,regroup2) = bin_bad_rate(df4,'col_map_bin',target)
            chi2 = cal_chi2(regroup2,all_bad_rate) # 计算后一个区间和bad01_bin的卡方值
            if chi1<chi2:  # 当chi1<chi2时,删除前一个区间对应的cutoff点
                cutoffpoints.remove(cutoffpoints[bad01_bin_index-1])
            else:  # 当chi1>=chi2时,删除bin01对应的cutoff点
                cutoffpoints.remove(cutoffpoints[bad01_bin_index])
        df2['col_map_bin'] = df2['col_map'].apply(lambda x:assign_bin(x,cutoffpoints))
        (dict_bad,regroup) = bin_bad_rate(df2,'col_map_bin',target)
        # 重新将col_map映射至区间，并计算最小和最大的违约率，直达不再出现违约率为0或1的情况，循环停止
        [min_bad_rate,max_bad_rate] = [min(dict_bad.values()),max(dict_bad.values())]
    
    # 检查分箱后的最小占比
    if min_binpct>0:
        group_values = df2['col_map'].apply(lambda x:assign_bin(x,cutoffpoints))
        df2['col_map_bin'] = group_values # 将col_map映射为对应的区间Bin
        group_df = group_values.value_counts().to_frame() 
        group_df['bin_pct'] = group_df['col_map']/n # 计算每个区间的占比
        min_pct = group_df.bin_pct.min() # 得出最小的区间占比
        while min_pct<min_binpct and len(cutoffpoints)>2: # 当最小的区间占比小于min_pct且cutoff点的个数大于2，执行循环
            # 下面的逻辑基本与“检验是否有箱体只有好/坏样本”的一致
            min_pct_index = group_df[group_df.bin_pct==min_pct].index.tolist()
            min_pct_bin = min_pct_index[0]
            if min_pct_bin == max(group_df.index):
                cutoffpoints=cutoffpoints[:-1]
            elif min_pct_bin == min(group_df.index):
                cutoffpoints=cutoffpoints[1:]
            else:
                minpct_bin_index = list(group_df.index).index(min_pct_bin)
                prev_pct_bin = list(group_df.index)[minpct_bin_index-1]
                df5 = df2[df2['col_map_bin'].isin([min_pct_bin,prev_pct_bin])]
                (dict_bad,regroup3) = bin_bad_rate(df5,'col_map_bin',target)
                chi3 = cal_chi2(regroup3,all_bad_rate)
                later_pct_bin = list(group_df.index)[minpct_bin_index+1]
                df6 = df2[df2['col_map_bin'].isin([min_pct_bin,later_pct_bin])]
                (dict_bad,regroup4) = bin_bad_rate(df6,'col_map_bin',target)
                chi4 = cal_chi2(regroup4,all_bad_rate)
                if chi3<chi4:
                    cutoffpoints.remove(cutoffpoints[minpct_bin_index-1])
                else:
                    cutoffpoints.remove(cutoffpoints[minpct_bin_index])
    return cutoffpoints

# 数值型变量的分箱（卡方分箱）
def binning_num(df,target,col_list,max_bin=None,min_binpct=None):
    """
    df:数据集
    target:目标变量的字段名
    col_list:变量list集合
    max_bin:最大的分箱个数
    min_binpct:区间内样本所占总体的最小比
    
    return:
    bin_df :list形式，里面存储每个变量的分箱结果
    iv_value:list形式，里面存储每个变量的IV值
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total-bad
    all_odds = good/bad
    inf = float('inf')
    ninf = float('-inf')
    bin_df=[]
    iv_value=[]
    for col in col_list:
        cut = ChiMerge(df,col,target,max_bin=max_bin,min_binpct=min_binpct)
        cut.insert(0,ninf)
        cut.append(inf)
        bucket = pd.cut(df[col],cut)
        d1 = df.groupby(bucket)
        d2 = pd.DataFrame()
        d2['min_bin'] = d1[col].min()
        d2['max_bin'] = d1[col].max()
        d2['total'] = d1[target].count()
        d2['totalrate'] = d2['total']/total
        d2['bad'] = d1[target].sum()
        d2['badrate'] = d2['bad']/d2['total']
        d2['good'] = d2['total'] - d2['bad']
        d2['goodrate'] = d2['good']/d2['total']
        d2['badattr'] = d2['bad']/bad
        d2['goodattr'] = (d2['total']-d2['bad'])/good
        d2['odds'] = d2['good']/d2['bad']
        GB_list=[]
        for i in d2.odds:
            if i>=all_odds:
                GB_index = str(round((i/all_odds)*100,0))+str('G')
            else:
                GB_index = str(round((all_odds/i)*100,0))+str('B')
            GB_list.append(GB_index)
        d2['GB_index'] = GB_list
        d2['woe'] = np.log(d2['badattr']/d2['goodattr'])
        d2['bin_iv'] = (d2['badattr']-d2['goodattr'])*d2['woe']
        d2['IV'] = d2['bin_iv'].sum()
        iv = d2['bin_iv'].sum().round(3)
        print('变量名:{}'.format(col))
        print('IV:{}'.format(iv))
        print('\t')
        bin_df.append(d2)
        iv_value.append(iv)
    return bin_df,iv_value


# 数值型变量的iv明细表
def iv_num(df,target,col_list,max_bin=None,min_binpct=None):
    """
    df:数据集
    target:目标变量的字段名
    col_list:变量list集合
    max_bin:最大的分箱个数
    min_binpct:区间内样本所占总体的最小比
    
    return :变量的iv明细表
    """
    bin_df,iv_value = binning_num(df,target,col_list,max_bin=max_bin,min_binpct=min_binpct)
    iv_df = pd.DataFrame({'col':col_list,
                          'iv':iv_value})
    iv_df = iv_df.sort_values('iv',ascending=False)
    return iv_df


# 自定义分箱
def binning_self(df,col,target,cut=None,right_border=True):
    """
    df: 数据集
    col:分箱的单个变量名
    cut:划分区间的list
    right_border：设定左开右闭、左闭右开
    
    return: 
    bin_df: df形式，单个变量的分箱结果
    iv_value: 单个变量的iv
    """
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    all_odds = good/bad
    bucket = pd.cut(df[col],cut,right=right_border)
    d1 = df.groupby(bucket)
    d2 = pd.DataFrame()
    d2['min_bin'] = d1[col].min()
    d2['max_bin'] = d1[col].max()
    d2['total'] = d1[target].count()
    d2['totalrate'] = d2['total']/total
    d2['bad'] = d1[target].sum()
    d2['badrate'] = d2['bad']/d2['total']
    d2['good'] = d2['total'] - d2['bad']
    d2['goodrate'] = d2['good']/d2['total']
    d2['badattr'] = d2['bad']/bad
    d2['goodattr'] = (d2['total']-d2['bad'])/good
    d2['odds'] = d2['good']/d2['bad']
    GB_list=[]
    for i in d2.odds:
        if i>=all_odds:
            GB_index = str(round((i/all_odds)*100,0))+str('G')
        else:
            GB_index = str(round((all_odds/i)*100,0))+str('B')
        GB_list.append(GB_index)
    d2['GB_index'] = GB_list
    d2['woe'] = np.log(d2['badattr']/d2['goodattr'])
    d2['bin_iv'] = (d2['badattr']-d2['goodattr'])*d2['woe']
    d2['IV'] = d2['bin_iv'].sum()
    iv_value = d2['bin_iv'].sum().round(3)
    print('变量名:{}'.format(col))
    print('IV:{}'.format(iv_value))
    bin_df = d2.copy()
    return bin_df,iv_value


# 变量分箱结果的检查

# woe的可视化
def plot_woe(bin_df,hspace=0.4,wspace=0.4,plt_size=None,plt_num=None,x=None,y=None):
    """
    bin_df:list形式，里面存储每个变量的分箱结果
    hspace :子图之间的间隔(y轴方向)
    wspace :子图之间的间隔(x轴方向)
    plt_size :图纸的尺寸
    plt_num :子图的数量
    x :子图矩阵中一行子图的数量
    y :子图矩阵中一列子图的数量
    
    return :每个变量的woe变化趋势图
    """
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    for i,df in zip(range(1,plt_num+1,1),bin_df):
        col_name = df.index.name
        df = df.reset_index()
        plt.subplot(x,y,i)
        plt.title(col_name)
        sns.barplot(data=df,x=col_name,y='woe')
        plt.xlabel('')
        plt.xticks(rotation=30)
    return plt.show()


# 检验woe是否单调 
def woe_monoton(bin_df):
    """
    bin_df:list形式，里面存储每个变量的分箱结果
    
    return :
    woe_notmonoton_col :woe没有呈单调变化的变量，list形式
    woe_judge_df :df形式，每个变量的检验结果
    """
    woe_notmonoton_col =[]
    col_list = []
    woe_judge=[]
    for woe_df in bin_df:
        col_name = woe_df.index.name
        woe_list = list(woe_df.woe)
        if woe_df.shape[0]==2:
            #print('{}是否单调: True'.format(col_name))
            col_list.append(col_name)
            woe_judge.append('True')
        else:
            woe_not_monoton = [(woe_list[i]<woe_list[i+1] and woe_list[i]<woe_list[i-1])  or (woe_list[i]>woe_list[i+1] and woe_list[i]>woe_list[i-1]) for i in range(1,len(woe_list)-1,1)]
            if True in woe_not_monoton:
                #print('{}是否单调: False'.format(col_name))
                woe_notmonoton_col.append(col_name)
                col_list.append(col_name)
                woe_judge.append('False')
            else:
                #print('{}是否单调: True'.format(col_name))
                col_list.append(col_name)
                woe_judge.append('True')
    woe_judge_df = pd.DataFrame({'col':col_list,
                                 'judge_monoton':woe_judge})
    return woe_notmonoton_col,woe_judge_df


# 检查某个区间的woe是否大于1
def woe_large(bin_df):
    """
    bin_df:list形式，里面存储每个变量的分箱结果
    
    return:
    woe_large_col: 某个区间woe大于1的变量，list集合
    woe_judge_df :df形式，每个变量的检验结果
    """
    woe_large_col=[]
    col_list =[]
    woe_judge =[]
    for woe_df in bin_df:
        col_name = woe_df.index.name
        woe_list = list(woe_df.woe)
        woe_large = list(filter(lambda x:x>=1,woe_list))
        if len(woe_large)>0:
            col_list.append(col_name)
            woe_judge.append('True')
            woe_large_col.append(col_name)
        else:
            col_list.append(col_name)
            woe_judge.append('False')
    woe_judge_df = pd.DataFrame({'col':col_list,
                                 'judge_large':woe_judge})
    return woe_large_col,woe_judge_df

import numpy as np
import pandas as pd

def SplitData(df, col, numOfSplit, special_attribute=[]):
    '''
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    :return: 在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    '''
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = int(N/numOfSplit)
    splitPointIndex = [i*n for i in range(1,numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint # col中“切分点“右边第一个值



# def Chi2(df, total_col, bad_col, overallRate):
#     '''
#     :param df: 包含全部样本总计与坏样本总计的数据框
#     :param total_col: 全部样本的个数
#     :param bad_col: 坏样本的个数
#     :param overallRate: 全体样本的坏样本占比
#     :return: 卡方值
#     '''
#     df2 = df.copy()
#     # 期望坏样本个数＝全部样本个数*平均坏样本占比
#     df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
#     combined = zip(df2['expected'], df2[bad_col])
#     chi = [(i[0]-i[1])**2/i[0] for i in combined]
#     chi2 = sum(chi)
#     return chi2


def Chi2(df, total_col, bad_col):
    '''
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :return: 卡方值
    ''' 
    df2 = df.copy()
    # 求出df中，总体的坏样本率和好样本率
    badRate = sum(df2[bad_col])*1.0/sum(df2[total_col])
    # 当全部样本只有好或者坏样本时，卡方值为0
    if badRate in [0,1]:
        return 0
    df2['good'] = df2.apply(lambda x: x[total_col] - x[bad_col], axis = 1)
    goodRate = sum(df2['good']) * 1.0 / sum(df2[total_col])
    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df[total_col].apply(lambda x: x*badRate)
    df2['goodExpected'] = df[total_col].apply(lambda x: x * goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0]-i[1])**2/i[0] for i in badCombined]
    goodChi = [(i[0] - i[1]) ** 2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2


# Chi2 的另外一种计算方法
# def Chi2(df, total_col, bad_col):
#     df2 = df.copy()
#     df2['good'] = df2[total_col] - df2[bad_col]
#     goodTotal = sum(df2['good'])
#     badTotal = sum(df2[bad_col])
#     p1 = df2.loc[0]['good']*1.0/df2.loc[0][total_col]
#     p2 = df2.loc[1]['good']*1.0/df2.loc[1][total_col]
#     w1 = df2.loc[0]['good']*1.0/goodTotal
#     w2 = df2.loc[0][bad_col]*1.0/badTotal
#     N = sum(df2[total_col])
#     return N*(p1-p2)*(w1-w2)


def BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left') # 每箱的坏样本数，总样本数
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1) # 加上一列坏样本率
    dicts = dict(zip(regroup[col],regroup['bad_rate'])) # 每箱对应的坏样本率组成的字典
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)



### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge(df, col, target, max_interval=5,special_attribute=[],minBinPcnt=0):
    '''
    :param df: 包含目标变量与分箱属性的数据框
    :param col: 需要分箱的属性
    :param target: 目标变量，取值0或1
    :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数
    :param special_attribute: 不参与分箱的属性取值
    :param minBinPcnt：最小箱的占比，默认为0
    :return: 分箱结果
    '''
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  #如果原始属性的取值个数低于max_interval，不执行这段函数
        print ("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        if len(special_attribute)>=1:
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy() # 去掉special_attribute后的df
        N_distinct = len(list(set(df2[col])))

        # 步骤一: 通过col对数据集进行分组，求出每组的总样本数与坏样本数
        if N_distinct > 100:
            split_x = SplitData(df2, col, 100)
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
            # Assgingroup函数：每一行的数值和切分点做对比，返回原值在切分后的映射，
            # 经过map以后，生成该特征的值对象的“分箱”后的值
        else:
            df2['temp'] = df2[col]
        # 总体bad rate将被用来计算expected bad count
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)

        # 首先，每个单独的属性值将被分为单独的一组
        # 对属性值进行排序，然后两两组别进行合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels] #把每个箱的值打包成[[],[]]的形式

        # 步骤二：建立循环，不断合并最优的相邻两个组别，直到：
        # 1，最终分裂出来的分箱数<＝预设的最大分箱数
        # 2，每箱的占比不低于预设值（可选）
        # 3，每箱同时包含好坏样本
        # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        split_intervals = max_interval - len(special_attribute)
        while (len(groupIntervals) > split_intervals):  # 终止条件: 当前分箱数＝预设的分箱数
            # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
            chisqList = []
            for k in range(len(groupIntervals)-1):
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                #chisq = Chi2(df2b, 'total', 'bad', overallRate)
                chisq = Chi2(df2b, 'total', 'bad')
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            # 把groupIntervals的值改成类似的值改成类似从[[1][2],[3]]到[[1,2],[3]]
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined+1]
            groupIntervals.remove(groupIntervals[best_comnbined+1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]] #

        # 检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
        groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints)) #每个原始箱对应卡方分箱后的箱号
        df2['temp_Bin'] = groupedvalues
        (binBadRate,regroup) = BinBadRate(df2, 'temp_Bin', target)
        #返回（每箱坏样本率字典，和包含“列名、坏样本数、总样本数、坏样本率的数据框”）
        [minBadRate, maxBadRate] = [min(binBadRate.values()),max(binBadRate.values())]
        while minBadRate ==0 or maxBadRate == 1:
            # 找出全部为好／坏样本的箱
            indexForBad01 = regroup[regroup['bad_rate'].isin([0,1])].temp_Bin.tolist()
            bin=indexForBad01[0]
            # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
            else:
                # 和前一箱进行合并，并且计算卡方值
                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                #chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                chisq1 = Chi2(df2b, 'total', 'bad')
                # 和后一箱进行合并，并且计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex + 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                #chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                chisq2 = Chi2(df2b, 'total', 'bad')
                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])
            # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]
        # 需要检查分箱后的最小占比
        if minBinPcnt > 0:
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            valueCounts = groupedvalues.value_counts().to_frame()
            valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / sum(valueCounts['temp']))
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])
            while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
                # 找出占比最小的箱
                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
                # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                else:
                    # 和前一箱进行合并，并且计算卡方值
                    currentIndex = list(valueCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                    #chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                    chisq1 = Chi2(df2b, 'total', 'bad')
                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                    #chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                    chisq2 = Chi2(df2b, 'total', 'bad')
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])
                groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
                df2['temp_Bin'] = groupedvalues
                valueCounts = groupedvalues.value_counts().to_frame()
                valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / sum(valueCounts['temp']))
                valueCounts = valueCounts.sort_index()
                minPcnt = min(valueCounts['pcnt'])
        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints



def UnsupervisedSplitBin(df,var,numOfSplit = 5, method = 'equal freq'):
    '''
    :param df: 数据集
    :param var: 需要分箱的变量。仅限数值型。
    :param numOfSplit: 需要分箱个数，默认是5
    :param method: 分箱方法，'equal freq'：，默认是等频，否则是等距
    :return:
    '''
    if method == 'equal freq':
        N = df.shape[0]
        n = N / numOfSplit
        splitPointIndex = [i * n for i in range(1, numOfSplit)]
        rawValues = sorted(list(df[col]))
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
        return splitPoint
    else:
        var_max, var_min = max(df[var]), min(df[var])
        interval_len = (var_max - var_min)*1.0/numOfSplit
        splitPoint = [var_min + i*interval_len for i in range(1,numOfSplit)]
        return splitPoint



def AssignGroup(x, bin):
    '''
    :param x: 某个变量的某个取值
    :param bin: 上述变量的分箱结果
    :return: x在分箱结果下的映射
    '''
    N = len(bin)
    if x<=min(bin):
        return min(bin)
    elif x>max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]


def BadRateEncoding(df, col, target):
    '''
    :param df: dataframe containing feature and target
    :param col: the feature that needs to be encoded with bad rate, usually categorical type
    :param target: good/bad indicator
    :return: the assigned bad rate to encode the categorical feature
    '''
    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding':badRateEnconding, 'bad_rate':br_dict}


def AssignBin(x, cutOffPoints,special_attribute=[]):
    '''
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)



def CalcWOE(df, col, target):
    '''
    :param df: 包含需要计算WOE的变量和目标变量
    :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
    :param target: 目标变量，0、1表示好、坏
    :return: 返回WOE和IV
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV':IV}



## 判断某变量的坏样本率是否单调
def BadRateMonotone(df, sortByVar, target,special_attribute = []):
    '''
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateNotMonotone = [badRate[i]<badRate[i+1] and badRate[i] < badRate[i-1] or badRate[i]>badRate[i+1] and badRate[i] > badRate[i-1]
                       for i in range(1,len(badRate)-1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True

def MergeBad0(df,col,target, direction='bad'):
    '''
     :param df: 包含检验0％或者100%坏样本率
     :param col: 分箱后的变量或者类别型变量。检验其中是否有一组或者多组没有坏样本或者没有好样本。如果是，则需要进行合并
     :param target: 目标变量，0、1表示好、坏
     :return: 合并方案，使得每个组里同时包含好坏样本
     '''
    regroup = BinBadRate(df, col, target)[1]
    if direction == 'bad':
        # 如果是合并0坏样本率的组，则跟最小的非0坏样本率的组进行合并
        regroup = regroup.sort_values(by = 'bad_rate')
    else:
        # 如果是合并0好样本样本率的组，则跟最小的非0好样本率的组进行合并
        regroup = regroup.sort_values(by='bad_rate',ascending=False)
    regroup.index = range(regroup.shape[0])
    col_regroup = [[i] for i in regroup[col]]
    del_index = []
    for i in range(regroup.shape[0]-1):
        col_regroup[i+1] = col_regroup[i] + col_regroup[i+1]
        del_index.append(i)
        if direction == 'bad':
            if regroup['bad_rate'][i+1] > 0:
                break
        else:
            if regroup['bad_rate'][i+1] < 1:
                break
    col_regroup2 = [col_regroup[i] for i in range(len(col_regroup)) if i not in del_index]
    newGroup = {}
    for i in range(len(col_regroup2)):
        for g2 in col_regroup2[i]:
            newGroup[g2] = 'Bin '+str(i)
    return newGroup

def Prob2Score(prob, basePoint, PDO):
    #将概率转化成分数且为正整数
    y = np.log(prob/(1-prob))
    return int(basePoint+PDO/np.log(2)*(-y))


### 计算KS值
def KS(df, score, target):
    '''
    :param df: 包含目标变量与预测值的数据集
    :param score: 得分或者概率
    :param target: 目标变量
    :return: KS值
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score,ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    return max(KS)



def MergeByCondition(x,condition_list):
    #condition_list是条件列表。满足第几个condition，就输出几
    s = 0
    for condition in condition_list:
        if eval(str(x)+condition):
            return s
        else:
            s+=1
    return s

