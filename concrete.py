#Nueral network for strength of concrete
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import seaborn as sns

#importing data
data = pd.read_csv("C://Users//SrinidhiR//Desktop//EXELR//assignments//nueral network//concrete.csv")

#exploratory data analysis
data.head()
data.tail()
data.shape
data.mean
data.isnull().sum() #there are no null values.

data.describe()
data.columns

data.nunique()   #unique values of all variables.

#visualisation
corelation = data.corr()
sns.heatmap(corelation,xticklabels=corelation.columns,yticklabels=corelation.columns,annot =True)

sns.pairplot(data)

sns.relplot(x="age",y ="strength",hue='strength',data = data)


#splitting data into training and test
train,test = train_test_split(data,test_size=0.2,random_state=45)

#dropping the size_category column from trainX and trainY and adding them to testX,testY
trainX = train.drop(['strength'],axis =1)
trainY= train['strength']
testX = test.drop(['strength'],axis =1)
testY = test['strength']

#model building
model = Sequential()
model.add(Dense(50,input_dim=3,activation="relu"))
model.add(Dense(40,input_dim=3,activation ="relu"))
model.add(Dense(20,input_dim=3,activation="relu"))
model.add(Dense(1,kernel_initializer= "normal",activation ="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

first_model = model

#fitting the array with epoch=10
first_model=first_model.fit(np.array(trainX),np.array(trainY),epochs=100) 

#predicting test data
pred_test = first_model.pred(np.array(trainX))

