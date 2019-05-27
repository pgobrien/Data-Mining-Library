from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt


# Data Processing


print("Retrieving Data and assigning headers...")
headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "goal"]

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, names=headers, na_values=np.nan)


print("Replacing ? with NaN")
df = df.replace(' ?', np.nan)
print(df.shape)
print("Removing rows with missing values")
df = df.dropna()

print("Factorizing....")
df['workclass'] = pd.factorize(df['workclass'])[0]
df['education'] = pd.factorize(df['education'])[0]
df['marital-status'] = pd.factorize(df['marital-status'])[0]
df['occupation'] = pd.factorize(df['occupation'])[0]
df['relationship'] = pd.factorize(df['relationship'])[0]
df['race'] = pd.factorize(df['race'])[0]
df['sex'] = pd.factorize(df['sex'])[0]
df['native-country'] = pd.factorize(df['native-country'])[0]
# <=50K is 0 and >50K is 1
df['goal'] = pd.factorize(df['goal'])[0]


print('Saving label...')
labels = df.loc[:, 'goal']

print('Removing label from Dataframe...')
df = df.drop(['goal'], axis=1)


# Start of the model
neigh = KNeighborsClassifier(n_neighbors=3)

# data and label
X = df.values
y = labels

# create split of test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# create learning model object
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# predict
pred = knn.predict(X_test)



print("ACCURACY OF TEST: K = 3")
print(accuracy_score(y_test, pred))


odd_k = [i for i in range(50) if i % 2 != 0]

def ten_fold_cval(aList):
    cv_scores = []
    for k in aList:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        print("K = " + str(k))
        print(str(scores.mean()))
        cv_scores.append(scores.mean())

    return cv_scores


cross_scores = ten_fold_cval(odd_k)


MSE = [1 - x for x in cross_scores]

optimal_k = odd_k[MSE.index(min(MSE))]

print("Optimal number of k is " + str(optimal_k))



plt.plot(odd_k, MSE)
plt.xlabel("Number of K")
plt.ylabel("Missclassification Error")
plt.show()






#kay_folds = KFold(n_splits=5)
#kay_folds.get_n_splits(X)



# for train_index, test_index in kay_folds.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     neigh.fit(X_train, y_train)
#     accuracy_score(neigh.predict(X_test), y_test)
#










