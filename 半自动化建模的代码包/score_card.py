import numpy as np
import pandas as pd
import math
# 数据预处理

# 每个变量缺失率的计算
def missing_cal(df):
    """
    df :数据集
    
    return：每个变量的缺失率
    """
    missing_series = df.isnull().sum()/df.shape[0]
    missing_df = pd.DataFrame(missing_series).reset_index()
    missing_df = missing_df.rename(columns={'index':'col',
                                            0:'missing_pct'})
    missing_df = missing_df.sort_values('missing_pct',ascending=False).reset_index(drop=True)
    return missing_df

# 变量的缺失分布图
def plot_missing_var(df,plt_size=None):
    """
    df: 数据集
    plt_size :图纸的尺寸
    
    return: 缺失分布图（直方图形式)
    """
    missing_df = missing_cal(df)
    plt.figure(figsize=plt_size)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = missing_df['missing_pct']
    plt.hist(x=x,bins=np.arange(0,1.1,0.1),color='hotpink',ec='k',alpha=0.8)
    plt.ylabel('缺失值个数')
    plt.xlabel('缺失率')
    return plt.show()


# 单个样本的缺失分布
def plot_missing_user(df,plt_size=None):
    """
    df: 数据集
    plt_size: 图纸的尺寸
    
    return :缺失分布图（折线图形式）
    """
    missing_series = df.isnull().sum(axis=1)
    list_missing_num  = sorted(list(missing_series.values))
    plt.figure(figsize=plt_size)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(df.shape[0]),list_missing_num)
    plt.ylabel('缺失变量个数')
    plt.xlabel('sanples')
    return plt.show()


# 缺失值剔除（单个变量）
def missing_delete_var(df,threshold=None):
    """
    df:数据集
    threshold:缺失率删除的阈值
    
    return :删除缺失后的数据集
    """
    df2 = df.copy()
    missing_df = missing_cal(df)
    missing_col_num = missing_df[missing_df.missing_pct>=threshold].shape[0]
    missing_col = list(missing_df[missing_df.missing_pct>=threshold].col)
    df2 = df2.drop(missing_col,axis=1)
    print('缺失率超过{}的变量个数为{}'.format(threshold,missing_col_num))
    return df2


# 缺失值剔除（单个样本）
def missing_delete_user(df,threshold=None):
    """
    df:数据集
    threshold:缺失个数删除的阈值
    
    return :删除缺失后的数据集
    """
    df2 = df.copy()
    missing_series = df.isnull().sum(axis=1)
    missing_list = list(missing_series)
    missing_index_list = []
    for i,j in enumerate(missing_list):
        if j>=threshold:
            missing_index_list.append(i)
    df2 = df2[~(df2.index.isin(missing_index_list))]
    print('缺失变量个数在{}以上的用户数有{}个'.format(threshold,len(missing_index_list)))
    return df2


# 缺失值填充（类别型变量）
def fillna_cate_var(df,col_list,fill_type=None):
    """
    df:数据集
    col_list:变量list集合
    fill_type: 填充方式：众数/当做一个类别
    
    return :填充后的数据集
    """
    df2 = df.copy()
    for col in col_list:
        if fill_type=='class':
            df2[col] = df2[col].fillna('unknown')
        if fill_type=='mode':
            df2[col] = df2[col].fillna(df2[col].mode()[0])
    return df2


# 数值型变量的填充
# 针对缺失率在5%以下的变量用中位数填充
# 缺失率在5%--15%的变量用随机森林填充,可先对缺失率较低的变量先用中位数填充，在用没有缺失的样本来对变量作随机森林填充
# 缺失率超过15%的变量建议当做一个类别
def fillna_num_var(df,col_list,fill_type=None,filled_df=None):
    """
    df:数据集
    col_list:变量list集合
    fill_type:填充方式：中位数/随机森林/当做一个类别
    filled_df :已填充好的数据集，当填充方式为随机森林时 使用
    
    return:已填充好的数据集
    """
    df2 = df.copy()
    for col in col_list:
        if fill_type=='median':
            df2[col] = df2[col].fillna(df2[col].median())
        if fill_type=='class':
            df2[col] = df2[col].fillna(-999)
        if fill_type=='rf':
            rf_df = pd.concat([df2[col],filled_df],axis=1)
            known = rf_df[rf_df[col].notnull()]
            unknown = rf_df[rf_df[col].isnull()]
            x_train = known.drop([col],axis=1)
            y_train = known[col]
            x_pre = unknown.drop([col],axis=1)
            rf = RandomForestRegressor(random_state=0)
            rf.fit(x_train,y_train)
            y_pre = rf.predict(x_pre)
            df2.loc[df2[col].isnull(),col] = y_pre
    return df2


# 常变量/同值化处理
def const_delete(df,col_list,threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold:同值化处理的阈值
    
    return :处理后的数据集
    """
    df2 = df.copy()
    const_col = []
    for col in col_list:
        const_pct = df2[col].value_counts().iloc[0]/df2[df2[col].notnull()].shape[0]
        if const_pct>=threshold:
            const_col.append(col)
    df2 = df2.drop(const_col,axis=1)
    print('常变量/同值化处理的变量个数为{}'.format(len(const_col)))
    return df2


# 分类型变量的降基处理
def descending_cate(df,col_list,threshold=None):
    """
    df: 数据集
    col_list:变量list集合
    threshold:降基处理的阈值
    
    return :处理后的数据集
    """
    df2 = df.copy()
    for col in col_list:
        value_series = df[col].value_counts()/df[df[col].notnull()].shape[0]
        small_value = []
        for value_name,value_pct in zip(value_series.index,value_series.values):
            if value_pct<=threshold:
                small_value.append(value_name)
        df2.loc[df2[col].isin(small_value),col]='other'
    return df2

# EDA分析

# 类别型变量的分布
def plot_cate_var(df,col_list,hspace=0.4,wspace=0.4,plt_size=None,plt_num=None,x=None,y=None):
    """
    df:数据集
    col_list:变量list集合
    hspace :子图之间的间隔(y轴方向)
    wspace :子图之间的间隔(x轴方向)
    plt_size :图纸的尺寸
    plt_num :子图的数量
    x :子图矩阵中一行子图的数量
    y :子图矩阵中一列子图的数量
    
    return :变量的分布图（柱状图形式）
    """
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i,col in zip(range(1,plt_num+1,1),col_list):
        plt.subplot(x,y,i)
        plt.title(col)
        sns.countplot(data=df,y=col)
        plt.ylabel('')
    return plt.show()


# 数值型变量的分布
def plot_num_col(df,col_list,hspace=0.4,wspace=0.4,plt_type=None,plt_size=None,plt_num=None,x=None,y=None):
    """
    df:数据集
    col_list:变量list集合
    hspace :子图之间的间隔(y轴方向)
    wspace :子图之间的间隔(x轴方向)
    plt_type: 选择直方图/箱线图
    plt_size :图纸的尺寸
    plt_num :子图的数量
    x :子图矩阵中一行子图的数量
    y :子图矩阵中一列子图的数量
    
    return :变量的分布图（箱线图/直方图）
    """
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if plt_type=='hist':
        for i,col in zip(range(1,plt_num+1,1),col_list):
            plt.subplot(x,y,i)
            plt.title(col)
            sns.distplot(df[col].dropna())
            plt.xlabel('')
    if plt_type=='box':
        for i,col in zip(range(1,plt_num+1,1),col_list):
            plt.subplot(x,y,i)
            plt.title(col)
            sns.boxplot(data=df,x=col)
            plt.xlabel('')
    return plt.show()


# 类别型变量的违约率分析
def plot_default_cate(df,col_list,target,hspace=0.4,wspace=0.4,plt_size=None,plt_num=None,x=None,y=None):
    """
    df:数据集
    col_list:变量list集合
    target ：目标变量的字段名
    hspace :子图之间的间隔(y轴方向)
    wspace :子图之间的间隔(x轴方向)
    plt_size :图纸的尺寸
    plt_num :子图的数量
    x :子图矩阵中一行子图的数量
    y :子图矩阵中一列子图的数量
    
    return :违约率分布图（柱状图形式）
    """
    all_bad = df[target].sum()
    total = df[target].count()
    all_default_rate = all_bad*1.0/total
    
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i,col in zip(range(1,plt_num+1,1),col_list):
        d1 = df.groupby(col)
        d2 = pd.DataFrame()
        d2['total'] = d1[target].count()
        d2['bad'] = d1[target].sum()
        d2['default_rate'] = d2['bad']/d2['total']
        d2 = d2.reset_index()
        plt.subplot(x,y,i)
        plt.title(col)
        plt.axvline(x=all_default_rate)
        sns.barplot(data=d2,y=col,x='default_rate')
        plt.ylabel('')
    return plt.show()


# 数值型变量的违约率分析
def plot_default_num(df,col_list,target,hspace=0.4,wspace=0.4,q=None,plt_size=None,plt_num=None,x=None,y=None):
    """
    df:数据集
    col_list:变量list集合
    target ：目标变量的字段名
    hspace :子图之间的间隔(y轴方向)
    wspace :子图之间的间隔(x轴方向)
    q :等深分箱的箱体个数
    plt_size :图纸的尺寸
    plt_num :子图的数量
    x :子图矩阵中一行子图的数量
    y :子图矩阵中一列子图的数量
    
    return :违约率分布图（折线图形式）
    """
    all_bad = df[target].sum()
    total = df[target].count()
    all_default_rate = all_bad*1.0/total 
    
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    for i,col in zip(range(1,plt_num+1,1),col_list):
        bucket = pd.qcut(df[col],q=q,duplicates='drop')
        d1 = df.groupby(bucket)
        d2 = pd.DataFrame()
        d2['total'] = d1[target].count()
        d2['bad'] = d1[target].sum()
        d2['default_rate'] = d2['bad']/d2['total']
        d2 = d2.reset_index()
        plt.subplot(x,y,i)
        plt.title(col)
        plt.axhline(y=all_default_rate)
        sns.pointplot(data=d2,x=col,y='default_rate',color='hotpink')
        plt.xticks(rotation=60)
        plt.xlabel('')
    return plt.show()


# coding: utf-8

# In[ ]:


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
        iv = d2['bin_iv'].sum().round(3)
        print('变量名:{}'.format(col))
        print('IV:{}'.format(iv))
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
            woe_not_monoton = [(woe_list[i]<woe_list[i+1] and woe_list[i]<woe_list[i-1])                                or (woe_list[i]>woe_list[i+1] and woe_list[i]>woe_list[i-1])                                for i in range(1,len(woe_list)-1,1)]
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


# coding: utf-8

# In[ ]:


# 变量woe离散化

# 变量woe结果表
def woe_df_concat(bin_df):
    """
    bin_df:list形式，里面存储每个变量的分箱结果
    
    return :woe结果表
    """
    woe_df_list =[]
    for df in bin_df:
        woe_df = df.reset_index().assign(col=df.index.name).rename(columns={df.index.name:'bin'})
        woe_df_list.append(woe_df)
    woe_result = pd.concat(woe_df_list,axis=0)
    # 为了便于查看，将字段名列移到第一列的位置上
    woe_result1 = woe_result['col']
    woe_result2 = woe_result.iloc[:,:-1]
    woe_result_df = pd.concat([woe_result1,woe_result2],axis=1)
    woe_result_df = woe_result_df.reset_index(drop=True)
    return woe_result_df

# woe转换
def woe_transform(df,target,df_woe):
    """
    df:数据集
    target:目标变量的字段名
    df_woe:woe结果表
    
    return:woe转化之后的数据集
    """
    df2 = df.copy()
    for col in df2.drop([target],axis=1).columns:
        x = df2[col]
        bin_map = df_woe[df_woe.col==col]
        bin_res = np.array([0]*x.shape[0],dtype=float)
        for i in bin_map.index:
            lower = bin_map['min_bin'][i]
            upper = bin_map['max_bin'][i]
            if lower == upper:
                x1 = x[np.where(x == lower)[0]]
            else:
                x1 = x[np.where((x>=lower)&(x<=upper))[0]]
            mask = np.in1d(x,x1)
            bin_res[mask] = bin_map['woe'][i]
        bin_res = pd.Series(bin_res,index=x.index)
        bin_res.name = x.name
        df2[col] = bin_res
    return df2


# coding: utf-8

# In[ ]:


# 变量筛选 

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


# coding: utf-8

# In[ ]:


# 模型评估

# AUC 
def plot_roc(y_label,y_pred):
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:ROC曲线
    """
    tpr,fpr,threshold = metrics.roc_curve(y_label,y_pred) 
    AUC = metrics.roc_auc_score(y_label,y_pred) 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tpr,fpr,color='blue',label='AUC=%.3f'%AUC) 
    ax.plot([0,1],[0,1],'r--')
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_title('ROC')
    ax.legend(loc='best')
    return plt.show(ax)


# KS 
def plot_model_ks(y_label,y_pred):
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:KS曲线
    """
    pred_list = list(y_pred) 
    label_list = list(y_label)
    total_bad = sum(label_list)
    total_good = len(label_list)-total_bad 
    items = sorted(zip(pred_list,label_list),key=lambda x:x[0]) 
    step = (max(pred_list)-min(pred_list))/200 
    
    pred_bin=[]
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201): 
        idx = min(pred_list)+i*step 
        pred_bin.append(idx) 
        label_bin = [x[1] for x in items if x[0]<idx] 
        bad_num = sum(label_bin)
        good_num = len(label_bin)-bad_num  
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(pred_bin,good_rate,color='green',label='good_rate')
    ax.plot(pred_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(pred_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)


# 交叉验证
def cross_verify(x,y,estimators,fold,scoring='roc_auc'):
    """
    x:自变量的数据集
    y:target的数据集
    estimators：验证的模型
    fold：交叉验证的策略
    scoring:评级指标，默认auc
    
    return:交叉验证的结果
    """
    cv_result = cross_val_score(estimator=estimators,X=x,y=y,cv=fold,n_jobs=-1,scoring=scoring)
    print('CV的最大AUC为:{}'.format(cv_result.max()))
    print('CV的最小AUC为:{}'.format(cv_result.min()))
    print('CV的平均AUC为:{}'.format(cv_result.mean()))
    plt.figure(figsize=(6,4))
    plt.title('交叉验证的评价指标分布图')
    plt.boxplot(cv_result,patch_artist=True,showmeans=True,
            boxprops={'color':'black','facecolor':'yellow'},
            meanprops={'marker':'D','markerfacecolor':'tomato'},
            flierprops={'marker':'o','markerfacecolor':'red','color':'black'},
            medianprops={'linestyle':'--','color':'orange'})
    return plt.show()


# 学习曲线
def plot_learning_curve(estimator,x,y,cv=None,train_size = np.linspace(0.1,1.0,5),plt_size =None):
    """
    estimator :画学习曲线的基模型
    x:自变量的数据集
    y:target的数据集
    cv:交叉验证的策略
    train_size:训练集划分的策略
    plt_size:画图尺寸
    
    return:学习曲线
    """
    from sklearn.model_selection import learning_curve
    train_sizes,train_scores,test_scores = learning_curve(estimator=estimator,
                                                          X=x,
                                                          y=y,
                                                          cv=cv,
                                                          n_jobs=-1,
                                                          train_sizes=train_size)
    train_scores_mean = np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores,axis=1)
    test_scores_mean = np.mean(test_scores,axis=1)
    test_scores_std = np.std(test_scores,axis=1)
    plt.figure(figsize=plt_size)
    plt.xlabel('Training-example')
    plt.ylabel('score')
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std,alpha=0.1,color='r')
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Training-score')
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='cross-val-score')
    plt.legend(loc='best')
    return plt.show()


# 混淆矩阵 /分类报告
def plot_matrix_report(y_label,y_pred): 
    """
    y_label:测试集的y
    y_pred:对测试集预测后的概率
    
    return:混淆矩阵
    """
    matrix_array = metrics.confusion_matrix(y_label,y_pred)
    plt.matshow(matrix_array, cmap=plt.cm.summer_r)
    plt.colorbar()

    for x in range(len(matrix_array)): 
        for y in range(len(matrix_array)):
            plt.annotate(matrix_array[x,y], xy =(x,y), ha='center',va='center')

    plt.xlabel('True label')
    plt.ylabel('Predict label')
    print(metrics.classification_report(y_label,y_pred))
    return plt.show()


# coding: utf-8

# In[ ]:


# 评分卡实现

# 评分卡刻度 
def cal_scale(score,odds,PDO,model):
    """
    odds：设定的坏好比
    score:在这个odds下的分数
    PDO: 好坏翻倍比
    model:逻辑回归模型
    
    return :A,B,base_score
    """
    B = 20/(np.log(odds)-np.log(2*odds))
    A = score-B*np.log(odds)
    base_score = A+B*model.intercept_[0]
    print('B: {:.2f}'.format(B))
    print('A: {:.2f}'.format(A))
    print('基础分为：{:.2f}'.format(base_score))
    return A,B,base_score


# 变量得分表
def score_df_concat(woe_df,model,B):
    """
    woe_df: woe结果表
    model:逻辑回归模型
    
    return:变量得分结果表
    """
    coe = list(model.coef_[0])
    columns = list(woe_df.col.unique())
    scores=[]
    for c,col in zip(coe,columns):
        score=[]
        for w in list(woe_df[woe_df.col==col].woe):
            s = round(c*w*B,0)
            score.append(s)
        scores.extend(score)
    woe_df['score'] = scores
    score_df = woe_df.copy()
    return score_df


# 分数转换 
def score_transform(df,target,df_score):
    """
    df:数据集
    target:目标变量的字段名
    df_score:得分结果表
    
    return:得分转化之后的数据集
    """
    df2 = df.copy()
    for col in df2.drop([target],axis=1).columns:
        x = df2[col]
        bin_map = df_score[df_score.col==col]
        bin_res = np.array([0]*x.shape[0],dtype=float)
        for i in bin_map.index:
            lower = bin_map['min_bin'][i]
            upper = bin_map['max_bin'][i]
            if lower == upper:
                x1 = x[np.where(x == lower)[0]]
            else:
                x1 = x[np.where((x>=lower)&(x<=upper))[0]]
            mask = np.in1d(x,x1)
            bin_res[mask] = bin_map['score'][i]
        bin_res = pd.Series(bin_res,index=x.index)
        bin_res.name = x.name
        df2[col] = bin_res
    return df2


# 得分的KS 
def plot_score_ks(df,score_col,target):
    """
    df:数据集
    target:目标变量的字段名
    score_col:最终得分的字段名
    """
    total_bad = df[target].sum()
    total_good = df[target].count()-total_bad
    score_list = list(df[score_col])
    target_list = list(df[target])
    items = sorted(zip(score_list,target_list),key=lambda x:x[0]) 
    step = (max(score_list)-min(score_list))/200 
    
    score_bin=[] 
    good_rate=[] 
    bad_rate=[] 
    ks_list = [] 
    for i in range(1,201):
        idx = min(score_list)+i*step 
        score_bin.append(idx) 
        target_bin = [x[1] for x in items if x[0]<idx]  
        bad_num = sum(target_bin)
        good_num = len(target_bin)-bad_num 
        goodrate = good_num/total_good 
        badrate = bad_num/total_bad
        ks = abs(goodrate-badrate) 
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks_list.append(ks)
        
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(score_bin,good_rate,color='green',label='good_rate')
    ax.plot(score_bin,bad_rate,color='red',label='bad_rate')
    ax.plot(score_bin,ks_list,color='blue',label='good-bad')
    ax.set_title('KS:{:.3f}'.format(max(ks_list)))
    ax.legend(loc='best')
    return plt.show(ax)


# PR曲线
def plot_PR(df,score_col,target,plt_size=None):
    """
    df:得分的数据集
    score_col:分数的字段名
    target:目标变量的字段名
    plt_size:绘图尺寸
    
    return: PR曲线
    """
    total_bad = df[target].sum()
    score_list = list(df[score_col])
    target_list = list(df[target])
    score_unique_list = sorted(set(list(df[score_col])))
    items = sorted(zip(score_list,target_list),key=lambda x:x[0]) 

    precison_list = []
    tpr_list = []
    for score in score_unique_list:
        target_bin = [x[1] for x in items if x[0]<=score]  
        bad_num = sum(target_bin)
        total_num = len(target_bin)
        precison = bad_num/total_num
        tpr = bad_num/total_bad
        precison_list.append(precison)
        tpr_list.append(tpr)
    
    plt.figure(figsize=plt_size)
    plt.title('PR曲线')
    plt.xlabel('查全率')
    plt.ylabel('精确率')
    plt.plot(tpr_list,precison_list,color='tomato',label='PR曲线')
    plt.legend(loc='best')
    return plt.show()


# 得分分布图
def plot_score_hist(df,target,score_col,plt_size=None,cutoff=None):
    """
    df:数据集
    target:目标变量的字段名
    score_col:最终得分的字段名
    plt_size:图纸尺寸
    cutoff :划分拒绝/通过的点
    
    return :好坏用户的得分分布图
    """    
    plt.figure(figsize=plt_size)
    x1 = df[df[target]==1][score_col]
    x2 = df[df[target]==0][score_col]
    sns.kdeplot(x1,shade=True,label='坏用户',color='hotpink')
    sns.kdeplot(x2,shade=True,label='好用户',color ='seagreen')
    plt.axvline(x=cutoff)
    plt.legend()
    return plt.show()




# 得分明细表 
def score_info(df,score_col,target,x=None,y=None,step=None):
    """
    df:数据集
    target:目标变量的字段名
    score_col:最终得分的字段名
    x:最小区间的左值
    y:最大区间的右值
    step:区间的分数间隔
    
    return :得分明细表
    """
    df['score_bin'] = pd.cut(df[score_col],bins=np.arange(x,y,step),right=True)
    total = df[target].count()
    bad = df[target].sum()
    good = total - bad
    
    group = df.groupby('score_bin')
    score_info_df = pd.DataFrame()
    score_info_df['用户数'] = group[target].count()
    score_info_df['坏用户'] = group[target].sum()
    score_info_df['好用户'] = score_info_df['用户数']-score_info_df['坏用户']
    score_info_df['违约占比'] = score_info_df['坏用户']/score_info_df['用户数']
    score_info_df['累计用户'] = score_info_df['用户数'].cumsum()
    score_info_df['坏用户累计'] = score_info_df['坏用户'].cumsum()
    score_info_df['好用户累计'] = score_info_df['好用户'].cumsum()
    score_info_df['坏用户累计占比'] = score_info_df['坏用户累计']/bad 
    score_info_df['好用户累计占比'] = score_info_df['好用户累计']/good
    score_info_df['累计用户占比'] = score_info_df['累计用户']/total 
    score_info_df['累计违约占比'] = score_info_df['坏用户累计']/score_info_df['累计用户']
    score_info_df = score_info_df.reset_index()
    return score_info_df


# 绘制提升图和洛伦兹曲线
def plot_lifting(df,score_col,target,bins=10,plt_size=None):
    """
    df:数据集，包含最终的得分
    score_col:最终分数的字段名
    target:目标变量名
    bins:分数划分成的等份数
    plt_size:绘图尺寸
    
    return:提升图和洛伦兹曲线
    """
    score_list = list(df[score_col])
    label_list = list(df[target])
    items = sorted(zip(score_list,label_list),key = lambda x:x[0])
    step = round(df.shape[0]/bins,0)
    bad = df[target].sum()
    all_badrate = float(1/bins)
    all_badrate_list = [all_badrate]*bins
    all_badrate_cum = list(np.cumsum(all_badrate_list))
    all_badrate_cum.insert(0,0)
    
    score_bin_list=[]
    bad_rate_list = []
    for i in range(0,bins,1):
        index_a = int(i*step)
        index_b = int((i+1)*step)
        score = [x[0] for x in items[index_a:index_b]]
        tup1 = (min(score),)
        tup2 = (max(score),)
        score_bin = tup1+tup2
        score_bin_list.append(score_bin)
        label_bin = [x[1] for x in items[index_a:index_b]]
        bin_bad = sum(label_bin)
        bin_bad_rate = bin_bad/bad
        bad_rate_list.append(bin_bad_rate)
    bad_rate_cumsum = list(np.cumsum(bad_rate_list))
    bad_rate_cumsum.insert(0,0)
    
    plt.figure(figsize=plt_size)
    x = score_bin_list
    y1 = bad_rate_list
    y2 = all_badrate_list
    y3 = bad_rate_cumsum
    y4 = all_badrate_cum
    plt.subplot(1,2,1)
    plt.title('提升图')
    plt.xticks(np.arange(bins)+0.15,x,rotation=90)
    bar_width= 0.3
    plt.bar(np.arange(bins),y1,width=bar_width,color='hotpink',label='score_card')
    plt.bar(np.arange(bins)+bar_width,y2,width=bar_width,color='seagreen',label='random')
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.title('洛伦兹曲线图')
    plt.plot(y3,color='hotpink',label='score_card')
    plt.plot(y4,color='seagreen',label='random')
    plt.xticks(np.arange(bins+1),rotation=0)
    plt.legend(loc='best')
    return plt.show()


# 设定cutoff点，衡量有效性
def rule_verify(df,col_score,target,cutoff):
    """
    df:数据集
    target:目标变量的字段名
    col_score:最终得分的字段名    
    cutoff :划分拒绝/通过的点
    
    return :混淆矩阵
    """
    df['result'] = df.apply(lambda x:30 if x[col_score]<=cutoff else 10,axis=1)
    TP = df[(df['result']==30)&(df[target]==1)].shape[0] 
    FN = df[(df['result']==30)&(df[target]==0)].shape[0] 
    bad = df[df[target]==1].shape[0] 
    good = df[df[target]==0].shape[0] 
    refuse = df[df['result']==30].shape[0] 
    passed = df[df['result']==10].shape[0] 
    
    acc = round(TP/refuse,3) 
    tpr = round(TP/bad,3) 
    fpr = round(FN/good,3) 
    pass_rate = round(refuse/df.shape[0],3) 
    matrix_df = pd.pivot_table(df,index='result',columns=target,aggfunc={col_score:pd.Series.count},values=col_score) 
    
    print('精确率:{}'.format(acc))
    print('查全率:{}'.format(tpr))
    print('误伤率:{}'.format(fpr))
    print('规则拒绝率:{}'.format(pass_rate))
    return matrix_df


# coding: utf-8

# In[ ]:


# 绘制变量的得分占比偏移图
def plot_var_shift(df,day_col,score_col,plt_size=None):
    """
    df:变量在一段时间内，每个区间上的得分
    day_col:时间的字段名（天）
    score_col:得分的字段名
    plt_size: 绘图尺寸
    
    return:变量区间得分的偏移图
    """
    day_list = sorted(set(list(df[day_col]))) 
    score_list = sorted(set(list(df[score_col])))
    # 计算每天各个区间得分的占比
    prop_day_list = []
    for day in day_list:
        prop_list = []
        for score in score_list:
            prop = df[(df[day_col]==day)&(df[score_col]==score)].shape[0]/df[df[day_col]==day].shape[0]
            prop_list.append(prop)
        prop_day_list.append(prop_list)
    
    # 将得分占比的转化为画图的格式
    sub_list = []
    for p in prop_day_list:
        p_cumsum = list(np.cumsum(p))
        p_cumsum = p_cumsum[:-1]
        p_cumsum.insert(0,0)
        bar1_list = [1]*int(len(p_cumsum))
        sub = [bar1_list[i]-p_cumsum[i] for i in range(len(p_cumsum))]
        sub_list.append(sub)
    array = np.array(sub_list)
    
    stack_prop_list = [] # 面积图的y值
    bar_prop_list = [] # 堆积柱状图的y
    for i in range(len(score_list)):
        bar_prop = array[:,i]
        bar_prop_list.append(bar_prop)
        stack_prop = []
        for j in bar_prop:
            a = j
            b = j
            stack_prop.append(a)
            stack_prop.append(b)
        stack_prop_list.append(stack_prop)
    
    # 画图的x坐标轴
    x_bar = list(range(1,len(day_list)*2,2)) # 堆积柱状图的x值
    x_stack = []    # 面积图的x值
    for i in x_bar:
        c = i-0.5
        d = i+0.5
        x_stack.append(c)
        x_stack.append(d)
    
    # 绘图
    fig = plt.figure(figsize=plt_size)
    ax1 = fig.add_subplot(1,1,1)
    # 先清除x轴的刻度
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(''.format)) 
    ax1.set_xticks(range(1,len(day_list)*2,2))
    # 将y轴的刻度设置为百分比形式
    def to_percent(temp, position):
        return '%1.0f'%(100*temp) + '%'
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(to_percent))
    # 自定义x轴刻度标签
    for a,b in zip(x_bar,day_list): 
        ax1.text(a,-0.08,b,ha='center',va='bottom')
    # 绘制面积图和堆积柱状图
    for i,s in zip(range(len(day_list)),score_list):
        ax1.stackplot(x_stack,stack_prop_list[i],alpha=0.25)
        ax1.bar(x_bar,bar_prop_list[i],width=1,label='得分:{}'.format(s))
        # 添加y轴刻度虚线
        ax1.grid(True, 'major', 'y', ls='--', lw=.5, c='black', alpha=.3)
        ax1.legend(loc='best')
    plt.show()

    
# 计算评分的PSI
def score_psi(df1,df2,id_col,score_col,x,y,step=None):
    """
    df1:建模样本的得分,包含用户id,得分
    df2:上线样本的得分，包含用户id，得分
    id_col:用户id字段名
    score_col:得分的字段名
    x:划分得分区间的left值
    y:划分得分区间的right值
    step:步长
    
    return: 得分psi表
    """
    df1['score_bin'] = pd.cut(df1[score_col],bins=np.arange(x,y,step))
    model_score_group = df1.groupby('score_bin',as_index=False)[id_col].count().                           assign(pct=lambda x:x[id_col]/x[id_col].sum()).                           rename(columns={id_col:'建模样本户数',
                                           'pct':'建模户数占比'})
    df2['score_bin'] = pd.cut(df2[score_col],bins=np.arange(x,y,step))
    online_score_group = df2.groupby('score_bin',as_index=False)[id_col].count().                           assign(pct=lambda x:x[id_col]/x[id_col].sum()).                           rename(columns={id_col:'线上样本户数',
                                           'pct':'线上户数占比'})
    score_compare = pd.merge(model_score_group,online_score_group,on='score_bin',how='inner')
    score_compare['占比差异'] = score_compare['线上户数占比'] - score_compare['建模户数占比']
    score_compare['占比权重'] = np.log(score_compare['线上户数占比']/score_compare['建模户数占比'])
    score_compare['Index']= score_compare['占比差异']*score_compare['占比权重']
    score_compare['PSI'] = score_compare['Index'].sum()
    return score_compare


# 评分比较分布图
def plot_score_compare(df,plt_size=None):
    fig = plt.figure(figsize=plt_size)
    x = df.score_bin
    y1 = df.建模户数占比
    y2 = df.线上户数占比
    width=0.3
    plt.title('评分分布对比图')
    plt.xlabel('得分区间')
    plt.ylabel('用户占比')
    plt.xticks(np.arange(len(x))+0.15,x)
    plt.bar(np.arange(len(y1)),y1,width=width,color='seagreen',label='建模样本')
    plt.bar(np.arange(len(y2))+width,y2,width=width,color='hotpink',label='上线样本')
    plt.legend()
    return plt.show() 


# 变量稳定度分析
def var_stable(score_result,df,var,id_col,score_col,bins):
    """
    score_result:评分卡的score明细表，包含区间，用户数，用户占比,得分
    var：分析的变量名
    df:上线样本变量的得分，包含用户id,变量的value，变量的score
    id_col:df的用户id字段名
    score_col:df的得分字段名
    bins:变量划分的区间
    
    return :变量的稳定性分析表
    """
    model_var_group = score_result.loc[score_result.col==var,                      ['bin','total','totalrate','score']].reset_index(drop=True).                      rename(columns={'total':'建模用户数',
                                      'totalrate':'建模用户占比',
                                      'score':'得分'})
    df['bin'] = pd.cut(df[score_col],bins=bins)
    online_var_group = df.groupby('bin',as_index=False)[id_col].count()                         .assign(pct=lambda x:x[id_col]/x[id_col].sum())                         .rename(columns={id_col:'线上用户数',
                                          'pct':'线上用户占比'})
    var_stable_df = pd.merge(model_var_group,online_var_group,on='bin',how='inner')
    var_stable_df = var_stable_df.iloc[:,[0,3,1,2,4,5]]
    var_stable_df['得分'] = var_stable_df['得分'].astype('int64')
    var_stable_df['建模样本权重'] = np.abs(var_stable_df['得分']*var_stable_df['建模用户占比'])
    var_stable_df['线上样本权重'] = np.abs(var_stable_df['得分']*var_stable_df['线上用户占比'])
    var_stable_df['权重差距'] = var_stable_df['线上样本权重'] - var_stable_df['建模样本权重']
    return var_stable_df



