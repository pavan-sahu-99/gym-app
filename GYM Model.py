

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel(r"C:\Users\user\OneDrive\Desktop\Gymapp\dataGym.xlsx")

data['Class'] = LabelEncoder().fit_transform(data['Class'])


X =data.iloc[:,:3]
y = data.iloc[:,5:]


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








