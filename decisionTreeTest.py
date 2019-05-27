from sklearn import tree
from loadData import ImportData
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




# Data Processing
headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                   "hours-per-week", "native-country", "goal"]

csv = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
missing_value = ' ?'

adultData = ImportData(headers=headers, csv=csv, missing_value=missing_value, remove_goal=True)

# create split of test and train
X_train, X_test, y_train, y_test = train_test_split(adultData.X, adultData.y, test_size=0.25, random_state=42)


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


print(accuracy_score(y_test, pred))