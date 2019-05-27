import numpy as np
import pandas as pd


# current test data being used

headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                   "hours-per-week", "native-country", "goal"]

csv = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'



#TODO: Create into a proper class with methods instead of just a constructor

#TODO: Pass in option for mutiple classifiers with options to compare results

#TODO: Add methods to visualize the data

#TODO: Create classifier class that contains all the classifiers

class ImportData():

    def __init__(self, headers, csv, missing_value, remove_goal=True, limiter_150=None):
        print("Retrieving Data and assigning headers...")

        self.headers = headers
        self.remove_goal = remove_goal

        # for data vizualization to reduce points
        if limiter_150 == None or limiter_150 == True:
            self.df = pd.read_csv(csv, header=None, names=headers, na_values=np.nan, nrows=150)
        else:
            self.df = pd.read_csv(csv, header=None, names=headers, na_values=np.nan)

        print("Replacing ? with NaN")

        self.df = self.df.replace(missing_value, np.nan)
        print("Shape:")
        print(self.df.shape)
        print("Removing rows with missing values")
        self.df = self.df.dropna()

        #TODO: Add choice for one-hot encoding as label encoding can cause poor results
        # Factorize columns
        print("Factorizing...")
        for item in self.headers:
            if self.df[item].dtype == np.int64:
                continue
            else:
                self.df[item] = pd.factorize(self.df[item])[0]



        self.labels = self.df.iloc[:, -1]
        # retrieve the labels
        if self.remove_goal == True:
            print("Seperating labels from Data...")
            # remove labels from dataframe
            self.df = self.df.drop(self.headers[-1], axis=1)

        # Create variables for X and y to be consistent with terminology
        self.X = self.df.values
        self.y = self.labels




# TEST

myTest = ImportData(headers=headers,csv=csv, missing_value=' ?', remove_goal=False)

