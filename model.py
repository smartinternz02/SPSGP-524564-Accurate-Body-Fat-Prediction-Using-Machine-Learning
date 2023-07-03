# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:00:50 2023

@author: HOME
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#Load datset
df=pd.read_csv('bodyfat.csv')

print(df.head())
df['Bmi']=703*df['Weight']/(df['Height']*df['Height'])

#split to independent and dependent variables
X=df.drop(columns=['BodyFat'],axis=1)
Y=df['BodyFat']

#split to train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#feature scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#instantiate model
regressor=RandomForestRegressor()

#fit the model
regressor.fit(X_train,Y_train)

#make pickle file of model
import joblib
joblib.dump(regressor,open('random_forest_model.pkl','wb'))