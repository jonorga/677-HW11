###
### CS667 Data Science with Python, Homework 11, Jon Organ
###

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler


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

nb_acc = NB.score(X_test, Y_test)

print("2) Naive bayesian accuracy: " + str(round(NB.score(X_test, Y_test) * 100, 2)) + "%")

y_actu = pd.Series(Y_test.ravel(), name="Actual")
y_pred = pd.Series(predictions_nb, name="Predicted")
cm_nb = pd.crosstab(y_actu, y_pred)
print("3) Naive bayesian confusion matrix:")
print(cm_nb)


print("\n")
# Question 3 ====================================================================================================
print("Question 3:")

LR = LogisticRegression(solver='liblinear', random_state=0)
LR.fit(X_train, Y_train.ravel())

predictions_lr = LR.predict(X_test)
print("1) Logistic regression class labels predicted...")
y_pred = pd.Series(predictions_lr, name="Predicted")
cm_lr = pd.crosstab(y_actu, y_pred)

lr_acc = LR.score(X_test, Y_test)

print("2) Logistic regression accuracy: " + str(round(LR.score(X_test, Y_test) * 100, 2)) + "%")
print("3) Logistic regression confusion matrix:")
print(cm_lr)


print("\n")
# Question 4 ====================================================================================================
print("Question 4:")

DT = tree.DecisionTreeClassifier(criterion = 'entropy')
DT = DT.fit(X_train, Y_train)
predictions_dt = DT.predict(X_test)
print("1) Decision tree class labels predicted...")

y_pred = pd.Series(predictions_dt, name="Predicted")
cm_dt = pd.crosstab(y_actu, y_pred)

dt_acc = DT.score(X_test, Y_test)

print("2) Logistic regression accuracy: " + str(round(DT.score(X_test, Y_test) * 100, 2)) + "%")
print("3) Logistic regression confusion matrix:")
print(cm_dt)


print("\n")
# Question 5 ====================================================================================================
print("Question 5:")

N_vals = [1, 3, 5, 7, 9]
d_vals = [1, 2, 3, 4, 5]
rfc_scores = []


for N in N_vals:
	line = []
	for d in d_vals:
		model = RFC(n_estimators=N, max_depth=d, criterion='entropy')
		model.fit(X_train, Y_train.ravel())
		y_pred = model.predict(X_test)
		error_rate = zero_one_loss(Y_test, y_pred)
		rfc_scores.append([N, d, error_rate])

print("1) Random forest hyper-parameters run...")

plot_df = pd.DataFrame(rfc_scores, columns=['N', 'd', 'error_rate'])
scatter_plot = plt.figure()
ax = scatter_plot.add_subplot(1, 1, 1)

ax.scatter(plot_df["N"], plot_df["d"], s=plot_df['error_rate'] * 400)
ax.set_title("Scatter plot for random forest")
ax.set_xlabel("N value")
ax.set_ylabel("d value")
print("2) Saving random forest scatter plot...")
scatter_plot.savefig("Q5_scatterplot.png")

best_vals = plot_df.nsmallest(1, 'error_rate')
best_N = best_vals['N'].iloc[0]
best_d = best_vals['d'].iloc[0]
print("Best N value:", best_N)
print("Best d value:", best_d)

model = RFC(n_estimators=best_N, max_depth=best_d, criterion='entropy')
model.fit(X_train, Y_train.ravel())
rfc_acc = model.score(X_test, Y_test.ravel())
print("3) Random forest accuracy with highest performing hyper-parameters: " + 
	str(round(model.score(X_test, Y_test.ravel()) * 100, 2)) + "%")

predictions_rf = model.predict(X_test)
y_pred = pd.Series(predictions_rf, name="Predicted")
cm_rf = pd.crosstab(y_actu, y_pred)
print("4) Random forest confusion matrix:")
print(cm_rf)


print("\n")
# Question 6 ====================================================================================================
print("Question 6:")

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)

svm_lin = svm.SVC(kernel='linear')
svm_deg2 = svm.SVC(kernel='poly', degree=2)
svm_gauss = svm.SVC(kernel='rbf')

svm_lin.fit(X_train, Y_train.ravel())
svm_deg2.fit(X_train, Y_train.ravel())
svm_gauss.fit(X_train, Y_train.ravel())

predictions_svm_lin = svm_lin.predict(X_test)
predictions_svm_deg2 = svm_deg2.predict(X_test)
predictions_svm_gauss = svm_gauss.predict(X_test)
print("1) SVM class labels predicted...")

svm_lin_acc = svm_lin.score(X_test, Y_test)
svm_deg2_acc = svm_deg2.score(X_test, Y_test)
svm_gauss_acc = svm_gauss.score(X_test, Y_test)

print("2) SVM accuracy:")
print("SVM Linear: " + str(round(svm_lin.score(X_test, Y_test) * 100, 2)) + "%")
print("SVM Degree 2: " + str(round(svm_deg2.score(X_test, Y_test) * 100, 2)) + "%")
print("SVM Gaussian: " + str(round(svm_gauss.score(X_test, Y_test) * 100, 2)) + "%")

y_pred = pd.Series(predictions_svm_lin, name="Predicted")
cm_svm_lin = pd.crosstab(y_actu, y_pred)
y_pred = pd.Series(predictions_svm_deg2, name="Predicted")
cm_svm_deg2 = pd.crosstab(y_actu, y_pred)
y_pred = pd.Series(predictions_svm_gauss, name="Predicted")
cm_svm_gauss = pd.crosstab(y_actu, y_pred)


print("\n3) SVM confusion matrices:")
print("SVM Linear:")
print(cm_svm_lin)
print("\nSVM Degree 2:")
print(cm_svm_deg2)
print("\nSVM Gaussian:")
print(cm_svm_gauss)


print("\n")
# Question 7 ====================================================================================================
print("Question 7:")
print("{:<16} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('Method' ,'TP', 'FP', 'TN', 'FN', 
	'accuracy', 'TPR', 'TNR'))

def PrintTableLine(cm, clf, method, acc):
	TP = cm["0"].iloc[0]
	FP = cm["0"].iloc[1]
	TN = cm["1"].iloc[1]
	FN = cm["1"].iloc[0]
	TPR = round(TP / (TP + FN), 2)
	TNR = round(TN / (TN + FP), 2)
	print("{:<16} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(method ,TP, FP, TN, FN, 
		round(acc, 2), TPR, TNR))


PrintTableLine(cm_nb, NB, "Naive Bayesian", nb_acc)
PrintTableLine(cm_lr, LR, "Logistic", lr_acc)
PrintTableLine(cm_dt, DT, "Decision Tree", dt_acc)
PrintTableLine(cm_rf, model, "Random Forest", rfc_acc)
PrintTableLine(cm_svm_lin, svm_lin, "Linear SVM", svm_lin_acc)
PrintTableLine(cm_svm_deg2, svm_deg2, "Degree 2 SVM", svm_deg2_acc)
PrintTableLine(cm_svm_gauss, svm_gauss, "Gaussian SVM", svm_gauss_acc)


print("\n")
# Question 8 ====================================================================================================
print("Question 8:")
print("One way to find the importance of features is to create various test groups with different features "
	+ "and create confusion matrices from each run. The better the score from the testing groups, the more"
	+ " important the tested features are. Weights can be assigned based off the resulting scores.")




