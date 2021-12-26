import pandas as pd
import matplotlib.pyplot as plt #导入图像库
from sklearn.cross_validation import train_test_split

def outlier_processing(df,col):
    s=df[col]
    oneQuoter=s.quantile(0.25)
    threeQuote=s.quantile(0.75)
    irq=threeQuote-oneQuoter
    min=oneQuoter-1.5*irq
    max=threeQuote+1.5*irq
    df=df[df[col]<=max]
    df=df[df[col]>=min]
    return df

if __name__ == '__main__':
    data = pd.read_csv('MissingData.csv')
    # 年龄等于0的异常值进行剔除
    data = data[data['age'] > 0]
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]#剔除异常值
    data['SeriousDlqin2yrs']=1-data['SeriousDlqin2yrs']
    Y = data['SeriousDlqin2yrs']
    X = data.ix[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # print(Y_train)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
    train.to_csv('TrainData.csv',index=False)
    test.to_csv('TestData.csv',index=False)
    print(train.shape)
    print(test.shape)

