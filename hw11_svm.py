###
### CS667 Data Science with Python, Homework 11, Jon Organ
###

import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("cmg_weeks.csv")

# Question 1 ========================================================================================
print("Question 1:")

X_train = df[['Avg_Return', 'Volatility']][df['Week'] <= 50].values
Y_train = df[['Color']][df['Week'] <= 50].values
X_test = df[['Avg_Return', 'Volatility']][(df['Week'] > 50) & (df['Week'] <= 100)].values
Y_test = df[['Color']][(df['Week'] > 50) & (df['Week'] <= 100)].values

svm_lin = svm.SVC(kernel='linear')
svm_lin.fit(X_train, Y_train.ravel())
print("Year 2 SVM Linear accuracy: " + str(round(svm_lin.score(X_test, Y_test) * 100, 2)) + "%")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")