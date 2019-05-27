from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from loadData import ImportData

import matplotlib.pyplot as plt

#TODO: Clean up file


# Data Processing
headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                   "hours-per-week", "native-country", "goal"]

csv = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
missing_value = ' ?'

adultData = ImportData(headers=headers, csv=csv, missing_value=missing_value, remove_goal=True)

# create split of test and train
X_train, X_test, y_train, y_test = train_test_split(adultData.X, adultData.y, test_size=0.25, random_state=42)


logisticModel = LogisticRegression()

logisticModel.fit(X_train, y_train)
logisticModel.predict(X_test)

score = logisticModel.score(X_test, y_test)
print(score)

testCV = LogisticRegressionCV()
testCV.fit(adultData.X, adultData.y)