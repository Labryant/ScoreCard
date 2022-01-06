
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# 重新审视数据，对年和月进行分割，对不听的类别的特征添加更过的one-hot编码
# ont-hot
# 1. HY
# 2. ETYPE
# 3. ALEREO
# 4. WXCODE
# 5. 年和月（日期）
# 
# 
# > 可以确定的one-hot编码:
# + HY
# + ETYPE
# + CHANGE_FIRST_ALTERNO
# + CHANGE_END_ALTERNO
# + CHANGE_ALTERNO_MUCNID
# + RIGHT_FIRST_TYPECODE
# + RIGHT_END_TYPECODE
# + RIGHT_TYPECODE_MUCHID
# 
# 删除只有一个特征的脏特征。
# 
# 
# 对所有的有关时间还能的特征，分成年和月两个维度，
# 
# timeType = ['RGYEAR','FIRST_CHANGE_TIME','END_CHANGE_TIME','BRANCH_FIRST_YEAR','BRANCH_END_YEAR',
#             'BRANCH_FIRST_CLOSE_YEAR','TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME',
#            'RIGHT_FIRST_ASK_TIME', 'RIGHT_FIRST_FB_TIME','RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME',
#            'PROJECT_FIRST_TIME', 'PROJECT_END_TIME', 'LAWSUIT_FIRST_TIME', 'LAWSUIT_END_TIME',            
#             'BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME','BREAKFAITH_END_FIRST_TIME',
#            'RECRUIT_FIRST_TIME', 'RECRUIT_END_TIME']
# 
# 继续添加特征，
# 增加注册资本 / 2018.01 - RGYEAR。
# 增加注册资本注册RATE / 2018.01 - RGYEAR。


# In[2]:

# 所有时间类型的特征
timeType = ['RGYEAR','FIRST_CHANGE_TIME','END_CHANGE_TIME','BRANCH_FIRST_YEAR','BRANCH_END_YEAR',
            'BRANCH_FIRST_CLOSE_YEAR','TZ_QY_FIRST_TIME','TZ_QY_END_TIME','TZ_FIRST_CLOSE_TIME',
           'RIGHT_FIRST_ASK_TIME', 'RIGHT_FIRST_FB_TIME','RIGHT_END_ASK_TIME', 'RIGHT_END_FB_TIME',
           'PROJECT_FIRST_TIME', 'PROJECT_END_TIME', 'LAWSUIT_FIRST_TIME', 'LAWSUIT_END_TIME',            
            'BREAKFAITH_FIRST_FIRST_TIME', 'BREAKFAITH_FIRST_END_TIME','BREAKFAITH_END_FIRST_TIME',
           'RECRUIT_FIRST_TIME', 'RECRUIT_END_TIME','QUALIFICATION_FIRST_FIRST_TIME', 
            'QUALIFICATION_FIRST_END_TIME','QUALIFICATION_END_FIRST_TIME' ]


# 所有进行one-hot的编码的数据,对注册时间和变成时间得到的年和月进行one-hot编码
# one-hot的特征。
onehot = ['HY','ETYPE','CHANGE_FIRST_ALTERNO','CHANGE_END_ALTERNO','CHANGE_ALTERNO_MUCNID',
          'RIGHT_FIRST_TYPECODE','RIGHT_END_TYPECODE','RIGHT_TYPECODE_MUCHID','BREAKFAITH_FIRST_END_TIME_MONTH',
          'BREAKFAITH_FIRST_FIRST_TIME_MONTH','END_CHANGE_TIME_MONTH','FIRST_CHANGE_TIME_MONTH',
          'LAWSUIT_END_TIME_MONTH','LAWSUIT_END_TIME_YEAR','LAWSUIT_FIRST_TIME_MONTH','LAWSUIT_FIRST_TIME_YEAR',
          'PROJECT_END_TIME_MONTH','PROJECT_END_TIME_YEAR','PROJECT_FIRST_TIME_MONTH','PROJECT_FIRST_TIME_YEAR',
          'QUALIFICATION_END_FIRST_TIME_MONTH','QUALIFICATION_FIRST_END_TIME_MONTH','QUALIFICATION_FIRST_FIRST_TIME_MONTH',
          'RECRUIT_END_TIME_MONTH','RECRUIT_END_TIME_YEAR','RECRUIT_FIRST_TIME_MONTH','RECRUIT_FIRST_TIME_YEAR',
          'RIGHT_END_ASK_TIME_MONTH','RIGHT_END_FB_TIME_MONTH','RIGHT_FIRST_ASK_TIME_MONTH','RIGHT_FIRST_FB_TIME_MONTH']

def time2int(x):
    '''
    将时间转化为整数，加入训练
    2018-01 - x
    '''
    y = int(x[:x.find('-')])
    m = int(x[x.find('-')+1:])
    s = 12*(2018-y) + (1-m)
    return s


def removeOneValueFeature(df_all):
    '''
    删除只有一个值的特征，没有用处，而且占用空间,
    设置直接读取的特征文件。
    '''
    features = list(df_all.columns)
    for f in features:
        if len(set(df_all[f])) == 1:
            df_all.drop(f,axis=1,inplace=True)
    # 设置时间日期
    for t in timeType:
        df_all[t] = df_all[t].apply(time2int)

    df_all['XEID'] = df_all['EID'].map(lambda x: int(x[1:]))
    return df_all    

def runRemove(inputfile,outputfile):
    '''
    输入输出文件，文件名主义其格式的不同。
    '''
    df_all = pd.read_csv(inputfile)

    df_all = removeOneValueFeature(df_all)
    df_all.to_csv(outputfile,index=False,index_label=False)

    del df_all
    return outputfile

# 运行
inputfile = ['../data/alldata/df_data1234567890.csv','../data/alldata/df_data1234567890_plus.csv']
outputfile = ['../data/alldata/df_data_all.csv','../data/alldata/df_data_plus_all.csv']
for i in range(2):
    print(i," start ",inputfile[i])
    runRemove(inputfile[i],outputfile[i])
    print(i," end ",inputfile[i])

