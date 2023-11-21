###
### CS667 Data Science with Python, Homework 6, Jon Organ
###

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# Question 1 ====================================================================================================
print("Question 1:")
# Last digit of BUID: 6 (Group 3: MSTV, Width, Mode, Variance)
df = pd.read_excel("cardiotocography_data_set.xlsx", sheet_name=2, 
	usecols=["MSTV", "Width", "Mode", "Variance", "NSP"])
print("Excel file read...")

def AssignClasses(row):
	if row['NSP'] == 1:
		return "1"
	elif row['NSP'] == 2 or row['NSP'] == 3:
		return "0"
	else:
		return row['NSP']

df['NSP'] = df.apply(AssignClasses, axis=1)
print("Classes assigned...")
eodf = len(df.index)
df = df.drop(index=[0, eodf-3, eodf-2, eodf-1])

print("\n")
# Question 2 ====================================================================================================
print("Question 2:")

Y = df[["NSP"]].values
X = df[["MSTV", "Width", "Mode", "Variance"]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)

NB = GaussianNB()
NB.fit(X_train, Y_train.ravel())
predictions_nb = NB.predict(X_test)
print("1) Naive bayesian class labels predicted...")

print("2) Naive bayesian accuracy: " + str(round(NB.score(X_test, Y_test) * 100, 2)) + "%")

y_actu = pd.Series(Y_test.ravel(), name="Actual")
y_pred = pd.Series(predictions_nb, name="Predicted")
cm_nb = pd.crosstab(y_actu, y_pred)
print("3) Naive bayesian confusion matrix:")
print(cm_nb)


print("\n")
# Question 3 ====================================================================================================
print("Question 3:")



