
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # 得到所有的企业本身的特征
#     0. 企业注册时间差，RGYEAR
#     1. 2018-企业注册时间差，RGYEAR_DIFF
#     2. 注册资本，ZCZB
#     3. 企业的行业大类，HY
#     4. 企业的类型，ETYPE
#     5. 注册资本占整个行业大类的注册资本的平均值的比例，ZCZB_RATE
#     6. 注册资本占这个企业类型的注册资本的平均值的比例，ETYPE_RATE
#     
#     7. 已经完成的企业的计算指标，['MPNUM', 'INUM', 'FINZB','FSTINUM', 'TZINUM'],用0填充
#     -----------------------------------------------------------------------------------------
#     8. 每个企业对应行业大类的平均资金，HY_ZCZB_AVG
#     9. 每个企业对应行业大类的平均资金占所有企业的平均注册资金的比例, HY_ZCZB_AVG_RATE
# 
#     8. 每个企业对应行业大类的平均资金，ETYPE_ZCZB_AVG
#     9. 每个企业对应行业大类的平均资金占所有企业的平均注册资金的比例, ETYPE_ZCZB_AVG_RATE
#     
#         
#     10. 每个完成的企业的计算指标其对应行业大类的计算指标的平均值,['MPNUM_HY_AVG', 'INUM_HY_AVG', 'FINZB_HY_AVG','FSTINUM_HY_AVG', 'TZINUM_HY_AVG']
#     
#     
#         ['MPNUM_HY_AVG', 'INUM_HY_AVG', 'FINZB_HY_AVG','FSTINUM_HY_AVG', 'TZINUM_HY_AVG'] / MPNUM_AVG ... 
#     11. 每个完成的企业的计算指标其对应行业大类的计算指标的平均值占整个计算指标的平均值的比例,['MPNUM_HY_AVG_RATE', 'INUM_HY_AVG_RATE', 'FINZB_HY_AVG_RATE','FSTINUM_HY_AVG_RATE', 'TZINUM_HY_AVG_RATE']
#         
#         
#         ['MPNUM', 'INUM', 'FINZB','FSTINUM', 'TZINUM'] / ['MPNUM_HY_AVG', 'INUM_HY_AVG', 'FINZB_HY_AVG','FSTINUM_HY_AVG', 'TZINUM_HY_AVG']
#     12. 每个完成的企业的计算指标占其对应行业大类对应的计算指标的平均值的比例,['MPNUM_HY_RATE', 'INUM_HY_RATE', 'FINZB_HY_RATE','FSTINUM_HY_RATE', 'TZINUM_HY_RATE']
#     
#     ---------------------------------------------------------------------------------------
#     13. 每个完成的企业的计算指标其对应行业大类的计算指标的平均值,['MPNUM_ETYPE_AVG', 'INUM_ETYPE_AVG', 'FINZB_ETYPE_AVG','FSTINUM_ETYPE_AVG', 'TZINUM_ETYPE_AVG']
#     
#     
#         ['MPNUM_ETYPE_AVG', 'INUM_ETYPE_AVG', 'FINZB_ETYPE_AVG','FSTINUM_ETYPE_AVG', 'TZINUM_ETYPE_AVG'] / MPNUM_AVG ... 
#     14. 每个完成的企业的计算指标其对应行业大类的计算指标的平均值占整个计算指标的平均值的比例,['MPNUM_ETYPE_AVG_RATE', 'INUM_ETYPE_AVG_RATE', 'FINZB_ETYPE_AVG_RATE','FSTINUM_ETYPE_AVG_RATE', 'TZINUM_ETYPE_AVG_RATE']
#         
#         
#         ['MPNUM', 'INUM', 'FINZB','FSTINUM', 'TZINUM'] / ['MPNUM_ETYPE_AVG', 'INUM_ETYPE_AVG', 'FINZB_ETYPE_AVG','FSTINUM_ETYPE_AVG', 'TZINUM_ETYPE_AVG']
#     15. 每个完成的企业的计算指标占其对应行业大类对应的计算指标的平均值的比例,['MPNUM_ETYPE_RATE', 'INUM_ETYPE_RATE', 'FINZB_ETYPE_RATE','FSTINUM_ETYPE_RATE', 'TZINUM_ETYPE_RATE']
#     
#     
#     -------------------------------------------------------------------------------------
#     16. 添加对应HY和ETYPE最大值的比例
#     
#     10. 每个完成的企业的计算指标其对应行业大类的计算指标的最大值,['MPNUM_HY_MAX', 'INUM_HY_MAX', 'FINZB_HY_MAX','FSTINUM_HY_MAX', 'TZINUM_HY_MAX']
#     
#     
#         ['MPNUM_HY_MAX', 'INUM_HY_MAX', 'FINZB_HY_MAX','FSTINUM_HY_MAX', 'TZINUM_HY_MAX'] / MPNUM_MAX ... 
#     11. 每个完成的企业的计算指标其对应行业大类的计算指标的最大值值占整个计算指标的最大值值的比例,['MPNUM_HY_MAX_RATE', 'INUM_HY_MAX_RATE', 'FINZB_HY_MAX_RATE','FSTINUM_HY_MAX_RATE', 'TZINUM_HY_MAX_RATE']
#         
#         
#         ['MPNUM', 'INUM', 'FINZB','FSTINUM', 'TZINUM'] / ['MPNUM_HY_MAX', 'INUM_HY_MAX', 'FINZB_HY_MAX','FSTINUM_HY_MAX', 'TZINUM_HY_MAX']
#     12. 每个完成的企业的计算指标占其对应行业大类对应的计算指标的最大值值的比例,['MPNUM_HY_MAX_RATE', 'INUM_HY_MAX_RATE', 'FINZB_HY_MAX_RATE','FSTINUM_HY_MAX_RATE', 'TZINUM_HY_MAX_RATE']
#     
#     
# 
#     
# 

# In[2]:

company = pd.read_csv('../data/public/1entbase.csv')


# company['ETYPE'].value_counts()

company = company.fillna(0)
# company.info()
# company.head()


# In[6]:

# company['PROV'].value_counts()

# In[7]:

tmp = pd.DataFrame()
tmp['HY_ZCZB_AVG'] = company.groupby(['HY'],as_index=True,axis=0)['ZCZB'].mean()
tmp['HY_ZCZB_AVG_RATE'] = tmp['HY_ZCZB_AVG'] / company['ZCZB'].mean()

tmp['MPNUM_HY_AVG'] = company.groupby(['HY'],as_index=True,axis=0)['MPNUM'].mean()
tmp['INUM_HY_AVG'] = company.groupby(['HY'],as_index=True,axis=0)['INUM'].mean()
tmp['FINZB_HY_AVG'] = company.groupby(['HY'],as_index=True,axis=0)['FINZB'].mean()
tmp['FSTINUM_HY_AVG'] = company.groupby(['HY'],as_index=True,axis=0)['FSTINUM'].mean()
tmp['TZINUM_HY_AVG'] = company.groupby(['HY'],as_index=True,axis=0)['TZINUM'].mean()


tmp['HY_ZCZB_MAX'] = company.groupby(['HY'],as_index=True,axis=0)['ZCZB'].mean()
tmp['HY_ZCZB_MAX_RATE'] = tmp['HY_ZCZB_MAX'] / company['ZCZB'].mean()

tmp['MPNUM_HY_MAX'] = company.groupby(['HY'],as_index=True,axis=0)['MPNUM'].mean()
tmp['INUM_HY_MAX'] = company.groupby(['HY'],as_index=True,axis=0)['INUM'].mean()
tmp['FINZB_HY_MAX'] = company.groupby(['HY'],as_index=True,axis=0)['FINZB'].mean()
tmp['FSTINUM_HY_MAX'] = company.groupby(['HY'],as_index=True,axis=0)['FSTINUM'].mean()
tmp['TZINUM_HY_MAX'] = company.groupby(['HY'],as_index=True,axis=0)['TZINUM'].mean()


tmp['HY'] = tmp.index

tmp1 = pd.DataFrame()
tmp1['ETYPE_ZCZB_AVG'] = company.groupby(['ETYPE'],as_index=True,axis=0)['ZCZB'].mean()
tmp1['ETYPE_ZCZB_AVG_RATE'] = tmp1['ETYPE_ZCZB_AVG'] / company['ZCZB'].mean()

tmp1['MPNUM_ETYPE_AVG'] = company.groupby(['ETYPE'],as_index=True,axis=0)['MPNUM'].mean()
tmp1['INUM_ETYPE_AVG'] = company.groupby(['ETYPE'],as_index=True,axis=0)['INUM'].mean()
tmp1['FINZB_ETYPE_AVG'] = company.groupby(['ETYPE'],as_index=True,axis=0)['FINZB'].mean()
tmp1['FSTINUM_ETYPE_AVG'] = company.groupby(['ETYPE'],as_index=True,axis=0)['FSTINUM'].mean()
tmp1['TZINUM_ETYPE_AVG'] = company.groupby(['ETYPE'],as_index=True,axis=0)['TZINUM'].mean()

tmp1['ETYPE_ZCZB_MAX'] = company.groupby(['ETYPE'],as_index=True,axis=0)['ZCZB'].mean()
tmp1['ETYPE_ZCZB_MAX_RATE'] = tmp1['ETYPE_ZCZB_MAX'] / company['ZCZB'].mean()

tmp1['MPNUM_ETYPE_MAX'] = company.groupby(['ETYPE'],as_index=True,axis=0)['MPNUM'].mean()
tmp1['INUM_ETYPE_MAX'] = company.groupby(['ETYPE'],as_index=True,axis=0)['INUM'].mean()
tmp1['FINZB_ETYPE_MAX'] = company.groupby(['ETYPE'],as_index=True,axis=0)['FINZB'].mean()
tmp1['FSTINUM_ETYPE_MAX'] = company.groupby(['ETYPE'],as_index=True,axis=0)['FSTINUM'].mean()
tmp1['TZINUM_ETYPE_MAX'] = company.groupby(['ETYPE'],as_index=True,axis=0)['TZINUM'].mean()


tmp1['ETYPE'] = tmp1.index



company = pd.merge(company,tmp,how='left',on=['HY'])
company = pd.merge(company,tmp1,how='left',on=['ETYPE'])


# company.info()
# company.head()


# In[9]:

company['RGYEAR_DIFF'] = 2018 - company['RGYEAR']
company['ZCZB_RATE'] = company['ZCZB'] / company['HY_ZCZB_AVG']
company['ETYPE_RATE'] = company['ZCZB'] / company['ETYPE_ZCZB_AVG']

company['ZCZB_RATE_MAXN'] = company['ZCZB'] / company['HY_ZCZB_MAX']
company['ETYPE_RATE_MAXN'] = company['ZCZB'] / company['ETYPE_ZCZB_MAX']


# In[10]:

company['MPNUM_HY_AVG_RATE'] = company['MPNUM_HY_AVG'] / company['MPNUM'].mean()
company['INUM_HY_AVG_RATE'] = company['INUM_HY_AVG'] / company['INUM'].mean()
company['FINZB_HY_AVG_RATE'] = company['FINZB_HY_AVG'] / company['FINZB'].mean()
company['FSTINUM_HY_AVG_RATE'] = company['FSTINUM_HY_AVG'] / company['FSTINUM'].mean()
company['TZINUM_HY_AVG_RATE'] = company['TZINUM_HY_AVG'] / company['TZINUM'].mean()

company['MPNUM_HY_RATE'] = company['MPNUM'] / company['MPNUM_HY_AVG']
company['INUM_HY_RATE'] = company['INUM'] / company['INUM_HY_AVG']
company['FINZB_HY_RATE'] = company['FINZB'] / company['FINZB_HY_AVG']
company['FSTINUM_HY_RATE'] = company['FSTINUM'] / company['FSTINUM_HY_AVG']
company['TZINUM_HY_RATE'] = company['TZINUM'] / company['TZINUM_HY_AVG']



# In[11]:

company['MPNUM_HY_MAX_RATE'] = company['MPNUM_HY_MAX'] / company['MPNUM'].mean()
company['INUM_HY_MAX_RATE'] = company['INUM_HY_MAX'] / company['INUM'].mean()
company['FINZB_HY_MAX_RATE'] = company['FINZB_HY_MAX'] / company['FINZB'].mean()
company['FSTINUM_HY_MAX_RATE'] = company['FSTINUM_HY_MAX'] / company['FSTINUM'].mean()
company['TZINUM_HY_MAX_RATE'] = company['TZINUM_HY_MAX'] / company['TZINUM'].mean()

company['MPNUM_HY_MAX_RATE'] = company['MPNUM'] / company['MPNUM_HY_MAX']
company['INUM_HY_MAX_RATE'] = company['INUM'] / company['INUM_HY_MAX']
company['FINZB_HY_MAX_RATE'] = company['FINZB'] / company['FINZB_HY_MAX']
company['FSTINUM_HY_MAX_RATE'] = company['FSTINUM'] / company['FSTINUM_HY_MAX']
company['TZINUM_HY_MAX_RATE'] = company['TZINUM'] / company['TZINUM_HY_MAX']


# In[12]:

company['MPNUM_ETYPE_AVG_RATE'] = company['MPNUM_ETYPE_AVG'] / company['MPNUM'].mean()
company['INUM_ETYPE_AVG_RATE'] = company['INUM_ETYPE_AVG'] / company['INUM'].mean()
company['FINZB_ETYPE_AVG_RATE'] = company['FINZB_ETYPE_AVG'] / company['FINZB'].mean()
company['FSTINUM_ETYPE_AVG_RATE'] = company['FSTINUM_ETYPE_AVG'] / company['FSTINUM'].mean()
company['TZINUM_ETYPE_AVG_RATE'] = company['TZINUM_ETYPE_AVG'] / company['TZINUM'].mean()

company['MPNUM_ETYPE_RATE'] = company['MPNUM'] / company['MPNUM_ETYPE_AVG']
company['INUM_ETYPE_RATE'] = company['INUM'] / company['INUM_ETYPE_AVG']
company['FINZB_ETYPE_RATE'] = company['FINZB'] / company['FINZB_ETYPE_AVG']
company['FSTINUM_ETYPE_RATE'] = company['FSTINUM'] / company['FSTINUM_ETYPE_AVG']
company['TZINUM_ETYPE_RATE'] = company['TZINUM'] / company['TZINUM_ETYPE_AVG']



# In[13]:

company['MPNUM_ETYPE_MAX_RATE'] = company['MPNUM_ETYPE_MAX'] / company['MPNUM'].mean()
company['INUM_ETYPE_MAX_RATE'] = company['INUM_ETYPE_MAX'] / company['INUM'].mean()
company['FINZB_ETYPE_MAX_RATE'] = company['FINZB_ETYPE_MAX'] / company['FINZB'].mean()
company['FSTINUM_ETYPE_MAX_RATE'] = company['FSTINUM_ETYPE_MAX'] / company['FSTINUM'].mean()
company['TZINUM_ETYPE_MAX_RATE'] = company['TZINUM_ETYPE_MAX'] / company['TZINUM'].mean()

company['MPNUM_ETYPE_MAX_RATE'] = company['MPNUM'] / company['MPNUM_ETYPE_MAX']
company['INUM_ETYPE_MAX_RATE'] = company['INUM'] / company['INUM_ETYPE_MAX']
company['FINZB_ETYPE_MAX_RATE'] = company['FINZB'] / company['FINZB_ETYPE_MAX']
company['FSTINUM_ETYPE_MAX_RATE'] = company['FSTINUM'] / company['FSTINUM_ETYPE_MAX']
company['TZINUM_ETYPE_MAX_RATE'] = company['TZINUM'] / company['TZINUM_ETYPE_MAX']



# In[ ]:


# In[14]:

company = company.fillna(0)


# In[15]:

def setTime(x):
    return str(x)+'-01'

company['RGYEAR'] = company['RGYEAR'].apply(setTime)

# In[16]:

# company.info()
# company.head()


# HY的类别更多
# FSTINUM、INUM、MPNUM、FINZB:用同一个企业类型（HY）的平均值填充空值（取整）
# TZINUM：用这个HY中出现最多的值进行填充
# 

# In[17]:

company.to_csv('../data/alldata/df_data1.csv',index=False,index_label=False)


# In[ ]:



# In[ ]:



