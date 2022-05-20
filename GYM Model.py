

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel(r"C:\Users\user\OneDrive\Desktop\Gymapp\dataGym.xlsx")

#Data Cleaning
data['Class'] = data['Class'].replace('Healthy\xa0','Healthy').replace('EXtremely obese','Extremely obese').replace('Extremely obese','Extremelyobese')

l = ['Extremely obese--normal exercise+yoga+heavy dite control+TIPS ->Obese',
       'Healthy--fitness+protine+simple dite',
       'Obese--slow lean+protine+full dite ->Overweight',
       'Overweight--lean+cardio+protine+normal dite ->Healthy',
       'Under weight--protines+carbs+normal weights ->Healthy']
a = ['Extreme Calorie deficiet+ no cheat days +Cardio+ Regular Gym + Patience','Regular gym + Calorie surplus + keep it up',
     'Calorie deficiet+ Cardio+ Regular Gym+ Patience','Calorie deficiet+ Cardio+ Regular Gym', 'Calorie Surplus + Regular gym' ]
j = 0

for i in l:
    data['Prediction'] = data['Prediction'].replace(l[j],a[j])
    j+=1

#data['Class'] = LabelEncoder().fit_transform(data['Class'])


X =data.iloc[:,:3]
y = data.iloc[:,4]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

model_GYM = RandomForestClassifier(n_estimators=20)
model_GYM.fit(X_train, y_train.values.ravel())

print(model_GYM)


# make predictions
expected = y_test
predicted = model_GYM.predict(X_test)
# summarize the fit of the model
#Correction
metrics.classification_report(expected, predicted)
metrics.confusion_matrix(expected, predicted)

import pickle

pickle.dump(model_GYM, open("Model_GYM.pkl", "wb"))

model = pickle.load(open("Model_GYM.pkl", "rb"))

print(model.predict([[40,5.6,70]]))








