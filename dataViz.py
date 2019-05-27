import matplotlib.pyplot as plt
from loadData import ImportData

import seaborn as sns

# Data Processing


headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                   "hours-per-week", "native-country", "goal"]

csv = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
missing_value = ' ?'
data = ImportData(headers=headers, csv=csv, missing_value=missing_value, remove_goal=True, limiter_150=True)


class Viz():

    def __init__(self, headers, csv, missing_value, data, limiter_150=False):

        sns.set()

        self.headers = headers
        self.csv = csv
        self.missing_value = missing_value
        self.data = data
        self.limiter_150 = limiter_150



    def heatMap(self):
        corr = data.df.corr()
        sns.heatmap(self.data.df.corr(),
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)

        plt.show()


    def scatterPlotTwoFeatures(self, x_label, y_label):

        if self.limiter_150 == False:
            raise ValueError("Limiter_150 must be True")

        if not isinstance(x_label, str) or not isinstance(y_label, str):
            raise ValueError("x_Label and y_Label must be of type String")

        plt.scatter(self.data.df[x_label], self.data.df[y_label], c=["red", "blue"], label=["No", "Yes"])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def pairPlot(self):
        sns.pairplot(data.df)
        plt.show()


    def histo(self, col):
        sns.distplot(self.data.df[col])
        plt.show()

    def kernelDensity(self, col):
        sns.distplot(self.data.df[col], hist=False, rug=True)
        plt.show()




vizTest = Viz(headers=headers, csv=csv, missing_value=missing_value, data=data, limiter_150=True)
#vizTest.heatMap()