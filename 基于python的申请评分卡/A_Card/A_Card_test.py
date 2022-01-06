import pandas as pd
import re
import time
import datetime
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from numpy import log
from sklearn.metrics import roc_auc_score
import numpy as np
import scorecard_function
def CareerYear(x):
    #对工作年限进行转换
    x = str(x)
    if x.find('nan') > -1:
        return -1
    elif x.find("10+")>-1:   #将"10＋years"转换成 11
        return 11
    elif x.find('< 1') > -1:  #将"< 1 year"转换成 0
        return 0
    else:
        return int(re.sub("\D", "", x))   #其余数据，去掉"years"并转换成整数


def DescExisting(x):
    #将desc变量转换成有记录和无记录两种
    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConvertDateStr(x):
    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1','%Y-%m')))
        #time.mktime 不能读取1970年之前的日期
    else:
        yr = int(x[4:6])
        if yr <=17:
            yr = 2000+yr
        else:
            yr = 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr,mth,1)


def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate,earlyDate)
        yr = gap.years
        mth = gap.months
        return yr*12+mth
    else:
        return 0


def MakeupMissing(x):
    if np.isnan(x):
        return -1
    else:
        return x
folderOfData = '/Users/imxly/Desktop/apply/A_Card/'


def ModifyDf(x, new_value):
    if np.isnan(x):
        return new_value
    else:
        return x


'''
将模型应用在测试数据集上
'''

testDataFile = open(folderOfData+'testData.pkl','rb+')
testData = pickle.load((testDataFile))
testDataFile.close()

'''
第一步：完成数据预处理
'''

# 将带％的百分比变为浮点数
testData['int_rate_clean'] = testData['int_rate'].map(lambda x: float(x.replace('%',''))/100)

# 将工作年限进行转化，否则影响排序
testData['emp_length_clean'] = testData['emp_length'].map(CareerYear)

# 将desc的缺失作为一种状态，非缺失作为另一种状态
testData['desc_clean'] = testData['desc'].map(DescExisting)

# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
testData['app_date_clean'] = testData['issue_d'].map(lambda x: ConvertDateStr(x))
testData['earliest_cr_line_clean'] = testData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))

# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
testData['mths_since_last_delinq_clean'] = testData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))

testData['mths_since_last_record_clean'] = testData['mths_since_last_record'].map(lambda x:MakeupMissing(x))

testData['pub_rec_bankruptcies_clean'] = testData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))

'''
第二步：变量衍生
'''
# 考虑申请额度与收入的占比
testData['limit_income'] = testData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)

# 考虑earliest_cr_line到申请日期的跨度，以月份记
testData['earliest_cr_to_app'] = testData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)

'''
第三步：分箱并代入WOE值
'''
modelFile =open(folderOfData+'LR_Model_Normal.pkl','rb+')
LR = pickle.load(modelFile)
modelFile.close()

#对变量的处理只需针对入模变量即可
var_in_model = list(LR.pvalues.index)
var_in_model.remove('intercept')

file1 = open(folderOfData+'merge_bin_dict.pkl','rb+')
merge_bin_dict = pickle.load(file1)
file1.close()


file2 = open(folderOfData+'br_encoding_dict.pkl','rb+')
br_encoding_dict = pickle.load(file2)
file2.close()

file3 = open(folderOfData+'continous_merged_dict.pkl','rb+')
continous_merged_dict = pickle.load(file3)
file3.close()

file4 = open(folderOfData+'WOE_dict.pkl','rb+')
WOE_dict = pickle.load(file4)
file4.close()

for var in var_in_model:
    var1 = var.replace('_Bin_WOE','')

    # 有些取值个数少、但是需要合并的变量
    if var1 in merge_bin_dict.keys():
        print ("{} need to be regrouped".format(var1))
        testData[var1 + '_Bin'] = testData[var1].map(merge_bin_dict[var1])

    # 有些变量需要用bad rate进行编码
    if var1.find('_br_encoding')>-1:
        var2 =var1.replace('_br_encoding','')
        print ("{} need to be encoded by bad rate".format(var2))
        testData[var1] = testData[var2].map(br_encoding_dict[var2])
        #需要注意的是，有可能在测试样中某些值没有出现在训练样本中，从而无法得出对应的bad rate是多少。故可以用最坏（即最大）的bad rate进行编码
        max_br = max(testData[var1])
        testData[var1] = testData[var1].map(lambda x: ModifyDf(x, max_br))


    #上述处理后，需要加上连续型变量一起进行分箱
    if -1 not in set(testData[var1]):
        testData[var1+'_Bin'] = testData[var1].map(lambda x:scorecard_function.AssignBin(x, continous_merged_dict[var1]))
    else:
        testData[var1 + '_Bin'] = testData[var1].map(lambda x: scorecard_function.AssignBin(x, continous_merged_dict[var1], [-1]))

    #WOE编码
    var3 = var.replace('_WOE','')
    testData[var] = testData[var3].map(WOE_dict[var3])


'''
第四步：将WOE值代入LR模型，计算概率和分数
'''
testData['intercept'] = [1]*testData.shape[0]
#预测数据集中，变量顺序需要和LR模型的变量顺序一致
#例如在训练集里，变量在数据中的顺序是“负债比”在“借款目的”之前，对应地，在测试集里，“负债比”也要在“借款目的”之前
testData2 = testData[list(LR.params.index)]
testData['prob'] = LR.predict(testData2)

#计算KS和AUC
auc = roc_auc_score(testData['y'],testData['prob'])
ks = scorecard_function.KS(testData, 'prob', 'y')
print(ks, auc) #0.22147178455285899 0.646469049695


#%%
# file5 = open(folderOfData+'LR_classweight','rb+')
# WOE_dict = pickle.load(file5)
# file4.close()
# testData['prob'] = l1_logit.predict(testData2)
#
# #计算KS和AUC
# auc = roc_auc_score(testData['y'],testData['prob'])
# ks = scorecard_function.KS(testData, 'prob', 'y')
# print(ks, auc)
#%%
basePoint = 250
PDO = 200
testData['score'] = testData['prob'].map(lambda x:scorecard_function.Prob2Score(x, basePoint, PDO))
testData = testData.sort_values(by = 'score')
#print(testData['score'])

#画出分布图
plt.hist(testData['score'], 100)
plt.xlabel('score')
plt.ylabel('freq')
plt.title('distribution')
plt.show()



