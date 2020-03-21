import pandas as pd
from numpy import *
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import SKCompat
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score,precision_recall_curve, average_precision_score
import operator
import re
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder

#normalize the features using max-min to convert the values into [0,1] interval
def MaxMinNorm(df,col):
    ma, mi = max(df[col]), min(df[col])
    rangeVal = ma - mi
    if rangeVal == 0:
        print col
    df[col] = df[col].map(lambda x:(x-mi)*1.0/rangeVal)

def CareerYear(x):
    if not x==x:
        return -1
    #对工作年限进行转换
    #if x.find('n/a') > -1:
        #return -1
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
    if not x==x:
        return -1
    else:
        return x



'''
第一步：数据准备
'''
folderOfData = foldOfData = 'C:/Users/OkO/Desktop/Financial Data Analsys/3nd Series/Data/'
allData = pd.read_csv(folderOfData + 'application.csv',header = 0, encoding = 'latin1')
allData['term'] = allData['term'].apply(lambda x: int(x.replace(' months','')))
# 处理标签：Fully Paid是正常用户；Charged Off是违约用户
allData['y'] = allData['loan_status'].map(lambda x: int(x == 'Charged Off'))

'''
由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
'''
allData1 = allData.loc[allData.term == 36]
trainData, testData = train_test_split(allData1,test_size=0.4)



'''
第二步：数据预处理
'''
# 将带％的百分比变为浮点数
trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%',''))/100)
# 将工作年限进行转化，否则影响排序
trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)
# 将desc的缺失作为一种状态，非缺失作为另一种状态
trainData['desc_clean'] = trainData['desc'].map(DescExisting)
# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x))
trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))
# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))
trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x:MakeupMissing(x))
trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))

'''
第三步：变量衍生
'''
# 考虑申请额度与收入的占比
trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)
# 考虑earliest_cr_line到申请日期的跨度，以月份记
trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)


'''
对于类别型变量，需要onehot（独热）编码，再训练GBDT模型
'''
num_features = ['int_rate_clean','emp_length_clean','annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app','inq_last_6mths', \
                'mths_since_last_record_clean', 'mths_since_last_delinq_clean','open_acc','pub_rec','total_acc','limit_income','earliest_cr_to_app']
cat_features = ['home_ownership', 'verification_status','desc_clean', 'purpose', 'zip_code','addr_state','pub_rec_bankruptcies_clean']

v = DictVectorizer(sparse=False)
X1 = v.fit_transform(trainData[cat_features].to_dict('records'))
#将独热编码和数值型变量放在一起进行模型训练
X2 = matrix(trainData[num_features])
X = hstack([X1,X2])
Y = trainData['y']

x_train,x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)

#numnber of input layer nodes: dimension =
#number of hidden layer & number of nodes in them: hidden_units
#full link or not: droput. dropout = 1 means full link
#activation function: activation_fn. By default it is relu
#learning rate:

#Example: select the best number of units in the 1-layer hidden layer
#model_dir = path can make the next iteration starting from last termination
#define the DNN with 1 hidden layer
no_hidden_units_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for no_hidden_units in range(10,51,10):
    print("the current choise of hidden units number is {}".format(no_hidden_units))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,hidden_units=[no_hidden_units, no_hidden_units+10],n_classes=2,dropout = 0.5)
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 1000)
    #monitor the performance of the model using AUC score
    #clf_pred = clf._estimator.predict(x_test)
    #y_pred = [i for i in clf_pred]
    clf_pred_proba = clf._estimator.predict_proba(x_test)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_test,pred_proba)
    no_hidden_units_selection[no_hidden_units] = auc_score
best_hidden_units = max(no_hidden_units_selection.items(), key=lambda x: x[1])[0]


#Example: check the dropout effect
dropout_selection = {}
feature_columns = [tf.contrib.layers.real_valued_column("", dimension = x_train.shape[1])]
for dropout_prob in linspace(0,0.99,100):
    print("the current choise of drop out rate is {}".format(dropout_prob))
    clf0 = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
                                          hidden_units = [no_hidden_units],
                                          n_classes=2,
                                          dropout = dropout_prob
                                          #optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1,l1_regularization_strength=0.001
                                          #model_dir = path
                                          #learning_rate=0.1
                                          )
    clf = SKCompat(clf0)
    clf.fit(x_train, y_train, batch_size=256,steps = 10000)
    #monitor the performance of the model using AUC score
    #clf_pred = clf._estimator.predict(x_test)
    #y_pred = [i for i in clf_pred]
    clf_pred_proba = clf._estimator.predict_proba(x_test)
    pred_proba = [i[1] for i in clf_pred_proba]
    auc_score = roc_auc_score(y_test,pred_proba)
    dropout_selection[dropout_prob] = auc_score
best_dropout_prob = max(dropout_selection.iteritems(), key=operator.itemgetter(1))[0]
