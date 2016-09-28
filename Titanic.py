from __future__ import division
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import cross_validation


sns.set_style('whitegrid')

titanic_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(titanic_df.head())
print("----------------------------------------------------------------------------------------")
print(titanic_df.info())
print("----------------------------------------------------------------------------------------")
titanic_df["Age"] = titanic_df.fillna(titanic_df["Age"].median())
print("----------------------------------------------------------------------------------------")
#checking for columns with null values
print(titanic_df.isnull().any())
print("----------------------------------------------------------------------------------------")
#pre-processing data, dropping columns that will unlikely help us in predicting
titanic_df=titanic_df.drop(['PassengerId',"Ticket","Name"],axis=1)
test_df=test_df.drop(['PassengerId',"Ticket","Name"],axis=1)
titanic_df.loc[titanic_df["Sex"] =="male", "Sex"] = 0
titanic_df.loc[titanic_df["Sex"] =="female", "Sex"] = 1
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

titanic_df.loc[titanic_df["Embarked"]=="S","Embarked"]=0
titanic_df.loc[titanic_df["Embarked"]=="C","Embarked"]=1
titanic_df.loc[titanic_df["Embarked"]=="Q","Embarked"]=2

test_df.loc[titanic_df["Embarked"]=="S","Embarked"]=0
test_df.loc[titanic_df["Embarked"]=="C","Embarked"]=1
test_df.loc[titanic_df["Embarked"]=="Q","Embarked"]=2

print(test_df.head())

#plotting several parameters to visualise whether there is a correlation between these features and the predictions
sns.factorplot(x="Embarked",y="Survived",data=titanic_df,size=4,aspect=3)
sns.factorplot(x="Sex",y="Survived",data=titanic_df,size=4,aspect=3)

#training regression model
predictors = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

linReg = LinearRegression()
kf = KFold(titanic_df.shape[0],n_folds=3,random_state=1)
logisticReg = LogisticRegression(random_state = 1)
randFor = RandomForestClassifier(n_estimators = 25,random_state = 1)
linRegPredictions = []
logRegPredictions = []
randForPredictions = []

for train,test in kf:
	train_predictors = (titanic_df[predictors].iloc[train,:])
	train_target = titanic_df["Survived"].iloc[train]
	

	linReg.fit(train_predictors,train_target)
	logisticReg.fit(train_predictors,train_target)
	randFor.fit(train_predictors,train_target)
	
	linReg_Predictions = linReg.predict(titanic_df[predictors].iloc[test,:])
	logisticReg_Predictions = logisticReg.predict(titanic_df[predictors].iloc[test,:])
	randFor_Predictions = randFor.predict(titanic_df[predictors].iloc[test,:])

	linRegPredictions.append(linReg_Predictions)
	logRegPredictions.append(logisticReg_Predictions)
	randForPredictions.append(randFor_Predictions)

linRegPredictions = np.concatenate(linRegPredictions,axis =0)
logRegPredictions = np.concatenate(logRegPredictions,axis =0)
randForPredictions = np.concatenate(randForPredictions,axis = 0)

linRegPredictions [linRegPredictions > .5] =1
linRegPredictions [linRegPredictions <=.5] =0
#logRegPredictions [logRegPredictions > .5] =1
#logRegPredictions [logRegPredictions <=.5] =0

linRegAccuracy = sum(linRegPredictions[linRegPredictions == titanic_df["Survived"]])/len(linRegPredictions)
print("The accuracy for linear regression is : ",linRegAccuracy)

logRegAccuracy = sum(logRegPredictions[logRegPredictions == titanic_df["Survived"]])/len(linRegPredictions)
print("The accuracy for logistic regression is : ",logRegAccuracy)

randForAccuracy = sum(randForPredictions[randForPredictions == titanic_df["Survived"]])/len(randForPredictions)
print("The accuracy for random forest is : ",randForAccuracy)

