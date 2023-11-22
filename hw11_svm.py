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
svm_lin_acc = svm_lin.score(X_test, Y_test)
print("Year 2 SVM Linear accuracy: " + str(round(svm_lin_acc * 100, 2)) + "%")


print("\n")
# Question 2 ========================================================================================
print("Question 2:")

predictions_lin = svm_lin.predict(X_test)
y_actu = pd.Series(Y_test.ravel(), name="Actual")
y_pred = pd.Series(predictions_lin, name="Predicted")
cm = pd.crosstab(y_actu, y_pred)
print("Year 2 confusion matrix:")
print(cm)


print("\n")
# Question 3 ========================================================================================
print("Question 3:")

if len(cm.columns) == 1:
	if cm.columns[0] == "Red":
		TP = 0
		FP = 0
		TN = cm["Red"].iloc[1]
		FN = cm["Red"].iloc[0]
		TPR = round(TP / (TP + FN), 4)
		TNR = round(TN / (TN + FP), 4)
	else:
		TP = cm["Green"].iloc[0]
		FP = cm["Green"].iloc[1]
		TN = 0
		FN = 0
		TPR = round(TP / (TP + FN), 4)
		TNR = round(TN / (TN + FP), 4)
else:
	TP = cm["Green"].iloc[0]
	FP = cm["Green"].iloc[1]
	TN = cm["Red"].iloc[1]
	FN = cm["Red"].iloc[0]
	TPR = round(TP / (TP + FN), 4)
	TNR = round(TN / (TN + FP), 4)

print("Year 2 True positive rate: " + str(TPR * 100) + "%")
print("Year 2 True negative rate: " + str(TNR * 100) + "%")


print("\n")
# Question 4 ========================================================================================
print("Question 4:")
svm_gauss = svm.SVC(kernel='rbf')
svm_gauss.fit(X_train, Y_train.ravel())
svm_gauss_acc = svm_gauss.score(X_test, Y_test)

if svm_gauss_acc > svm_lin_acc:
	result = "higher"
else:
	result = "lower"
print("Gaussian SVM resulted in a " + result + " accuracy than linear for year 2")
print("Gaussian SVM year 2 accuracy: " + str(svm_gauss_acc * 100) + "%")


print("\n")
# Question 5 ========================================================================================
print("Question 5:")
svm_deg2 = svm.SVC(kernel='poly', degree=2)
svm_deg2.fit(X_train, Y_train.ravel())
svm_deg2_acc = svm_deg2.score(X_test, Y_test)

if svm_deg2_acc > svm_lin_acc:
	result = "higher"
else:
	result = "lower"
print("Polynomial SVM resulted in a " + result + " accuracy than linear for year 2")
print("Polynomial SVM year 2 accuracy: " + str(svm_deg2_acc * 100) + "%")


print("\n")
# Question 6 ========================================================================================
print("Question 6:")

def BNH(df):
	y2_start = df['Close'].iloc[50]
	y2_end = df['Close'].iloc[100]
	stock = 100 / y2_start
	return round(stock * y2_end, 2)


def SVMStrategy(df, actu, pred):
	i = 0
	balance = 100
	while i < 50:
		today_stock = balance / df['Close'].iloc[i + 50]
		tmr_stock = balance / df['Close'].iloc[i + 51]
		difference = abs(today_stock - tmr_stock)
		if actu[i] == pred[i]:
			balance += difference * df["Close"].iloc[i + 51]
		else:
			balance -= difference * df["Close"].iloc[i + 51]
		i += 1
	return round(balance, 2)

bnh_bal = BNH(df)
svm_bal = SVMStrategy(df, y_actu, y_pred)

if svm_bal > bnh_bal:
	result = "higher"
else:
	result = "lower"

print("For year 2 of the Chipotle stock, the SVM trading strategy ($" + str(svm_bal) + ") resulted in a "
	+ result + " balance at the end of the year than buy-and-hold ($" + str(bnh_bal) + ")")




