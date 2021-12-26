import pandas as pd
import matplotlib.pyplot as plt #导入图像库
from sklearn.ensemble import RandomForestRegressor

# 用随机森林对缺失值预测填充函数
def set_missing(df):
    # 把已有的数值型特征取出来
    process_df = df.ix[:,[5,0,1,2,3,4,6,7,8,9]]
    # 分成已知该特征和未知该特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].as_matrix()
    unknown = process_df[process_df.MonthlyIncome.isnull()].as_matrix()
    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200,max_depth=3,n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print(predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df

if __name__ == '__main__':
    #载入数据
    data = pd.read_csv('cs-training.csv')
    #数据集确实和分布情况
    data.describe().to_csv('DataDescribe.csv')#了解数据集的分布情况
    data=set_missing(data)#用随机森林填补比较多的缺失值
    data=data.dropna()#删除比较少的缺失值
    data = data.drop_duplicates()#删除重复项
    data.to_csv('MissingData.csv',index=False)
    data.describe().to_csv('MissingDataDescribe.csv')
    """
    #异常值处理
    #年龄等于0的异常值进行剔除
    data=data[data['age']>0]
    # 箱形图
    data379=data[['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse']]
    data379.boxplot()
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    data379 = data[['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']]
    #data379.boxplot()
    plt.show()
    #data.to_csv('PretreatmentData.csv')
    """