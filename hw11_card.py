###
### CS667 Data Science with Python, Homework 6, Jon Organ
###

import pandas as pd


# Question 1 ====================================================================================================
print("Question 1:")
# Last digit of BUID: 6 (Group 3: MSTV, Width, Mode, Variance)
df = pd.read_excel("cardiotocography_data_set.xlsx", sheet_name=2, 
	usecols=["MSTV", "Width", "Mode", "Variance", "NSP"])
print("Excel file read...")

def AssignClasses(row):
	if row['NSP'] == 1:
		return 1
	elif row['NSP'] == 2 or row['NSP'] == 3:
		return 0
	else:
		return row['NSP']

df['NSP'] = df.apply(AssignClasses, axis=1)
print("Classes assigned...")

print("\n")
# Question 2 ====================================================================================================
print("Question 2:")