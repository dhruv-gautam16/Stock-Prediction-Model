#we used RandomForestRegressor to predict the future prices of stock
#importing all the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Reading the datasets

test=pd.read_csv('../input/ai-hackathon-foobar-60-q2/stock_test.csv')
train=pd.read_csv('../input/ai-hackathon-foobar-60-q2/stock_train.csv')


#making a random column intializing it value to 0
train['FUTURE PRICES']=0
fid=train['Id']

#Converting Strings to Integer

l=LabelEncoder()
X=train.iloc[0:4].values
train.iloc[0:4]=l.fit_transform(X.astype('int64'))
X=train.iloc[0:1499].values
train.iloc[0:1499]=l.fit_transform(X.astype('int64'))

l=LabelEncoder()
X=test.iloc[0:3].values
test.iloc[0:3]=l.fit_transform(X.astype('int64'))
X=test.iloc[0:261].values
test.iloc[0:261]=l.fit_transform(X.astype('int64'))

#Taking the Training Variable

y_train=train['FUTURE PRICES']
x_train=train.drop(['FUTURE PRICES'],axis=1)

#Taking Variables for training

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.2,random_state=0)

pip=Pipeline([('scaler2',StandardScaler()),('random_test:',RandomForestRegressor())])

#Training the Model

pip.fit(x_train,y_train)

prediction=pip.predict(x_test)

acc=pip.score(x_test,y_test)

acc

predict=pip.predict(test)

#Converting the output to dataframe
output=pd.DataFrame({'Id':fid,'FUTURE PRICES':predict})

output.head()