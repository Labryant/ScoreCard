
# coding: utf-8

# ## 合并给定的结果文件，进行平均
# 

# In[1]:

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[38]:

def avg_result(results,alpha=0.215):
    """
    参数：
        results：结果列表（可以是Dataframe或者是file类型）
        alpha:概率阈值，默认是0.23,大于这个概率的是1
    
    return：
        平均之后的结果
    """
    if isinstance(results[0],str):
        for i in range(len(results)):
            print(i)
            results[i] = pd.read_csv(results[i])
    
    
    df_result = results[0]
    for i in range(1,len(results)):
        df_result['PROB'] = df_result['PROB'] + results[i]['PROB']
    
    df_result['PROB'] = df_result['PROB'] / len(results)
    
    df_result.loc[df_result['PROB']>=alpha,'FORTARGET'] = 1
    df_result.loc[df_result['PROB']<alpha,'FORTARGET'] = 0
    
    return df_result


# In[39]:

resultfiles = ['../xresult/xsubmussion_idstackfeature.csv',
               '../xresult/xsubmussion_plus_prov.csv',]


resultfiles1 = ['../xresult/sub_all.csv','../xresult/sub_all_prov.csv',
                '../xresult/sub_all_xgbstack.csv','../xresult/sub_onehot.csv',
                '../xresult/sub_onehot_xgbstack.csv','../xresult/sub_plus_all.csv',
                '../xresult/sub_plus_all_prov.csv']

# 设置两种文件的比例结果。
resultfiles2 = ['../xresult/sub_all_prov.csv','../xresult/sub_plus_all.csv',
                '../xresult/sub_plus_all_prov.csv']


# 全是第一个模型结果融合
resultfiles3 = ['../xresult/sub_all.csv',
                '../xresult/sub_all_xgbstack.csv','../xresult/sub_onehot.csv',
                '../xresult/sub_onehot_xgbstack.csv','../xresult/sub_plus_all.csv',]

# model_origin
outputfile1 = ['../xresult/sub_all.csv','../xresult/sub_plus_all.csv',
              '../xresult/sub_onehot.csv','../xresult/sub_plus_onehot.csv',
              '../xresult/sub_all_xgbstack.csv','../xresult/sub_plus_all_xgbstack.csv',
              '../xresult/sub_onehot_xgbstack.csv','../xresult/sub_plus_onehot_xgbstack.csv']

# model_avg_prov
outputfile2 = ['../xresult/sub_all_prov.csv','../xresult/sub_plus_all_prov.csv',
              '../xresult/sub_onehot_prov.csv','../xresult/sub_plus_onehot_prov.csv',
              '../xresult/sub_all_xgbstack_prov.csv','../xresult/sub_plus_all_xgbstack_prov.csv',
              '../xresult/sub_onehot_xgbstack_prov.csv','../xresult/sub_plus_onehot_xgbstack_prov.csv',
              '../xresult/sub_all_rfstack_prov.csv','../xresult/sub_plus_all_rfstack_prov.csv',
              '../xresult/sub_onehot_rfstack_prov.csv','../xresult/sub_plus_onehot_rfstack_prov.csv',
              '../xresult/sub_all_lgbstack_prov.csv','../xresult/sub_plus_all_lgbstack_prov.csv',
              '../xresult/sub_onehot_lgbstack_prov.csv','../xresult/sub_plus_onehot_lgbstack_prov.csv']

# 全部PROV的结果平均，58个
file1 = ['../xresult/sub_all.csv','../xresult/sub_plus_all.csv',
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
        '../xresult/sub_onehot_prov_lgbstack.csv','../xresult/sub_plus_onehot_prov_lgbstack.csv']


file2 = ['../xresult/sub_all_prov.csv','../xresult/sub_plus_all_prov.csv',
        '../xresult/sub_onehot_prov.csv','../xresult/sub_plus_onehot_prov.csv',
        '../xresult/sub_all_xgbstack_prov.csv','../xresult/sub_plus_all_xgbstack_prov.csv',
        '../xresult/sub_onehot_xgbstack_prov.csv','../xresult/sub_plus_onehot_xgbstack_prov.csv',
        '../xresult/sub_all_rfstack_prov.csv','../xresult/sub_plus_all_rfstack_prov.csv',
        '../xresult/sub_onehot_rfstack_prov.csv','../xresult/sub_plus_onehot_rfstack_prov.csv',
        '../xresult/sub_all_lgbstack_prov.csv','../xresult/sub_plus_all_lgbstack_prov.csv',
        '../xresult/sub_onehot_lgbstack_prov.csv','../xresult/sub_plus_onehot_lgbstack_prov.csv',
        '../xresult/sub_all_prov_xgbstack_prov.csv','../xresult/sub_plus_all_prov_xgbstack_prov.csv',
        '../xresult/sub_onehot_prov_xgbstack_prov.csv','../xresult/sub_plus_onehot_prov_xgbstack_prov.csv',
        '../xresult/sub_all_prov_rfstack_prov.csv','../xresult/sub_plus_all_prov_rfstack_prov.csv',
        '../xresult/sub_onehot_prov_rfstack_prov.csv','../xresult/sub_plus_onehot_prov_rfstack_prov.csv',
        '../xresult/sub_all_prov_lgbstack_prov.csv','../xresult/sub_plus_all_prov_lgbstack_prov.csv',
        '../xresult/sub_onehot_prov_lgbstack_prov.csv','../xresult/sub_plus_onehot_prov_lgbstack_prov.csv']


mergefile = ['../xresult/merge_sub_6.csv','../xresult/merge_sub_7.csv','../xresult/merge_sub_8.csv',
             '../xresult/merge_sub_10.csv','../xresult/merge_sub_11.csv','../xresult/merge_sub_12.csv',
             '../xresult/merge_sub_13.csv','../xresult/merge_sub_14.csv','../xresult/sub_dp_xgbstack.csv',
             '../xresult/merge_sub_all_1.csv','../xresult/merge_sub_final.csv','../xresult/finally_prov.csv']


# files = ['../xresult/merge_sub_17.csv' ,'../xresult/sub_example.csv']
# files.extend(file1)
# files.extend(file2)
# files.append('../xresult/merge_sub_10.csv')
# files.append('../xresult/sub_dp_xgbstack.csv')
# files.append('../xresult/sub_plus_all.csv')
# files.append('../xresult/sub_plus_onehot_prov_lgbstack_prov.csv')


files = []
files.extend(outputfile1)
files.extend(outputfile2)
files.extend(file1)
files.extend(file2)
files.extend(mergefile)


# In[40]:

df_result = avg_result(mergefile)


# In[44]:

df_result.to_csv('../xresult/xiaoran.csv',index=False,index_label=False)


# In[42]:

print(len(df_result[df_result['FORTARGET']==1]))
print(len(df_result[df_result['FORTARGET']==0]))

print("得到的测试集的正负样本的比例 = ", len(df_result[df_result['FORTARGET']==1]) / len(df_result[df_result['FORTARGET']==0]))
# In[43]:
print("训练集中正负样本的比例 = ",29680 / 188584)


# In[ ]:



