
# coding: utf-8

# In[1]:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn import preprocessing

from keras.optimizers import SGD
import pandas as pd


# In[4]:

def xtrain_and_test(df_all):
    '''
    得到训练数据和测试数据
    '''
    df_label = pd.read_csv('../data/public/train.csv')
    df_test_label = pd.read_csv('../data/public/evaluation_public.csv')
#     print(len(df_label))
#     print(len(df_test_label))
    df_label.drop('ENDDATE',axis=1,inplace=True)
    
    df_train = df_all[df_all['EID'].isin(df_label['EID'])]
    df_test = df_all[df_all['EID'].isin(df_test_label['EID'])]
    
    df_train = pd.merge(df_train,df_label,how='left',on=['EID'])
    
    return df_train,df_test



# In[ ]:




# In[9]:

df_all = pd.read_csv('../data/alldata/df_all_xgbstack.csv')
df_train,df_test = xtrain_and_test(df_all)

features = df_all.columns[0:]
features = list(features)
features.remove('EID')
label = 'TARGET'

len(features)


# In[10]:

x_train = preprocessing.scale(df_train[features])
x_test = preprocessing.scale(df_test[features])



# In[2]:

model = Sequential()


# In[11]:

model = Sequential()
model.add(Dense(1000, input_dim=605, activation='relu'))
model.add(Dropout(0.4))


model.add(Dense(600, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(400, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(40, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(10, activation='relu'))
model.add(Dropout(0.4))


model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-5, momentum=0.8, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# epochs = 40

model.fit(x_train, df_train[label],
          epochs=100, 
          batch_size=128,)


# In[ ]:

proba = model.predict_proba(x_test)



# In[ ]:

proba_test = pd.DataFrame()
proba_test['EID'] = df_test['EID']
proba_test['FORTARGET'] = [0 for i in range(len(df_test))]
proba_test['PROB'] = proba


# In[ ]:

proba_test.loc[proba_test['PROB']>=0.23,'FORTARGET'] = 1
proba_test.to_csv('../xresult/xsubmussion_xgbstack.csv',index=False,index_label=False)

print(len(proba_test[proba_test['FORTARGET']==1]))
print(len(proba_test[proba_test['FORTARGET']==0]))


