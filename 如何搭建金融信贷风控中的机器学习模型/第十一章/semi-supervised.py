import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from scipy.stats import multivariate_normal


##################################
#### 试验一：半监督聚类法生成标签 ####
##################################
'''
在该试验中，我们从已知标签的样本中选取一部分看成已知标签，剩余样本看成未知标签。
运用k-均值聚类法对未知标签的样本进行标签填补，再与真实标签做对比。
注意：本试验仅仅检验半监督聚类法的效果。真实环境中由于标签缺失，是无法进行对比的。
'''

def Semi_Supervised_KMean(x_good, x_bad, x_unlabeled):
    '''
    :param x_good, x_bad: 标签已知的正负样本
    :param x_unlabeled: 标签未知的样本，需要对其进行划分
    :return: x_unlabeled每个样本划分的类别
    '''
    cent_good_1 = x_good.mean(axis=0)
    cent_bad_1 = x_bad.mean(axis=0)
    N = x_unlabeled.shape[0]
    for i in range(10000):
        #每次迭代中，计算每一个样本距离当前簇中心的距离，按较小的距离进行类别划分
        cluster_good = x_good
        cluster_bad = x_bad
        label = [0] * N
        for j in range(N):
            dist1 = ((x_unlabeled[j,:].getA() -  cent_good_1.getA())**2).sum()
            dist2 = ((x_unlabeled[j,:].getA() -  cent_bad_1.getA())**2).sum()
            if dist2 < dist1:
                cluster_bad = np.vstack((cluster_bad, x_unlabeled[j,:]))
                label[j] = 1
            else:
                cluster_good = np.vstack((cluster_good, x_unlabeled[j, :]))
        #全部样本进行划分后，重新计算两个簇的几何中心
        cent_good_2 = cluster_good.mean(axis=0)
        cent_bad_2 = cluster_bad.mean(axis=0)
        if ((cent_good_1.getA() -  cent_good_2.getA())**2).sum() < 0.000001 and ((cent_bad_1.getA() -  cent_bad_2.getA())**2).sum() < 0.000001:
            break
        else:
            cent_good_1 = cent_good_2
            cent_bad_1 = cent_bad_2
    return label

#由于初始文件很大，故只读取前10000行数据做试验。且原数据中存在部分缺失，为简单起见只选择非缺失样本
data0 = pd.read_csv('/Users/Downloads/all/application_train.csv',header = 0, nrows = 10000)
data = data0[(data0['EXT_SOURCE_2'].notna()) & (data0['EXT_SOURCE_3'].notna())][['EXT_SOURCE_2','EXT_SOURCE_3','TARGET']]
good, bad = data[data['TARGET'] == 0], data[data['TARGET'] == 1]
del good['TARGET']
del bad['TARGET']

#检验两类样本在两个变量上的分布
sns.kdeplot(bad['EXT_SOURCE_2'])
sns.kdeplot(good['EXT_SOURCE_2'])
plt.legend(['default','non-default'])

sns.kdeplot(bad['EXT_SOURCE_3'])
sns.kdeplot(good['EXT_SOURCE_3'])
plt.legend(['default','non-default'])

#从已知标签的样本中选择10%作为已知标签，剩余的90%作为未知标签
good_data_label,good_data_unlabel = train_test_split(good,test_size = 0.9)
bad_data_label,bad_data_unlabel = train_test_split(bad,test_size = 0.9)

plt.scatter(bad_data_label['EXT_SOURCE_2'], bad_data_label['EXT_SOURCE_3'])
plt.scatter(good_data_label['EXT_SOURCE_2'], good_data_label['EXT_SOURCE_3'])
plt.legend(['default','non-default'])

data_unlabel = np.vstack((good_data_unlabel,bad_data_unlabel))
real_label = [0]*good_data_unlabel.shape[0] + [1]*bad_data_unlabel.shape[0]

#应用半监督k-均值聚类进行样本划分，并检验聚类的准确度
pred_label = Semi_Supervised_KMean(np.mat(good_data_label), np.mat(bad_data_label), np.mat(data_unlabel))
confusion_matrix(real_label, pred_label)


################################
#### 试验二：混合高斯法生成标签 ####
################################
'''
在该试验中，我们从违约样本中选取一部分看成标签缺失，与全部非违约样本组成标签未知的样本。
运用混合高斯法对未知标签的样本进行类别指派，再与真实标签做对比。
注意：本试验仅仅检验混合高斯法的效果。真实环境中由于标签缺失，是无法进行对比的。
'''

#读取数据，并将样本分为正例和负例两组
data0 = pd.read_csv('/Users/Downloads/all/application_train.csv',header = 0, nrows = 10000)
data = data0[(data0['EXT_SOURCE_2'].notna()) & (data0['EXT_SOURCE_3'].notna())][['EXT_SOURCE_2','EXT_SOURCE_3','TARGET']]
good, bad = data[data['TARGET'] == 0], data[data['TARGET'] == 1]

#随机从正例中抽取50%的样本，添加到负例中作为未标记样本
bad_labeled, bad_unlabeled = train_test_split(bad,test_size = 0.5)
df_unlabeled = pd.concat([bad_unlabeled, good])
real_label = [1]*bad_unlabeled.shape[0] + [0]*good.shape[0]

N_positive, N_unlabeled = bad_labeled.shape[0], df_unlabeled.shape[0]
N_total = N_positive + N_unlabeled
X_positive,X_unlabeled = np.matrix(bad_labeled), np.matrix(df_unlabeled)

#初始化参数
mu_positive, mu_negative = X_positive.mean(axis=0).A1, X_unlabeled.mean(axis=0).A1
S_positive, S_negative = np.cov(X_positive, rowvar=0),np.cov(X_unlabeled, rowvar=0)
a0 = a1 =0.5
gamma = np.matrix([[0.0]*2]*N_unlabeled)
for s in range(1000):
    #E步：根据当前模型参数计算未标记样本x_i属于各高斯混合成分的概率
    for j in range(N_unlabeled):
        p0 = multivariate_normal.pdf(X_unlabeled[j],mu_positive,S_positive)
        p1 = multivariate_normal.pdf(X_unlabeled[j],mu_negative,S_negative)
        gamma[j,0],gamma[j, 1] = a0 * p0 / (a0 * p0 + a1 * p1),a1 * p1 / (a0 * p0 + a1 * p1)
    #M步：基于gamma更新模型参数
    mu_positive = (gamma[:,0].T*X_unlabeled+X_positive.sum(axis=0))/(gamma[:,0].sum()+N_positive)
    mu_positive = mu_positive.A1
    mu_negative = gamma[:, 1].T * X_unlabeled / gamma[:, 1].sum()
    mu_negative = mu_negative.A1

    ss1 = 0
    for k in range(N_unlabeled):
        ss1 += gamma[k,0]*(X_unlabeled[k,:]-mu_positive).T*(X_unlabeled[k,:]-mu_positive)
    S_positive = (ss1 + (X_positive - mu_positive).T*(X_positive - mu_positive))/(gamma[:,0].sum()+N_positive)
    ss2 = 0
    for k in range(N_unlabeled):
        ss2 += gamma[k, 1] * (X_unlabeled[k, :] - mu_negative).T * (X_unlabeled[k, :] - mu_negative)
    S_negative = ss2 / gamma[:, 1].sum()

    a1 = (gamma[:,0].sum()+N_positive)/N_total
    a2 = gamma[:,1].sum()/N_total


##############################################
## 用混合高斯模型区分未标记样本中的正负样本 ##
##############################################
pred_label = [0]*N_unlabeled
for i in range(X_unlabeled.shape[0]):
    if gamma[i,0]>=gamma[i,1]:
        pred_label[i] = 1
confusion_matrix(real_label, pred_label)