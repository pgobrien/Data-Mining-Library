#TODO: add PCA
import seaborn as sns; sns.set()
from loadData import ImportData
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



headers = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                   "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                   "hours-per-week", "native-country", "goal"]

csv = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
missing_value = ' ?'
data = ImportData(headers=headers, csv=csv, missing_value=missing_value, remove_goal=True, limiter_150=True)
#df = pd.read_csv()


class testPCA():

    def __init__(self, data, headers, missing_values, n_components=None):

        self.data = data
        self.headers = headers
        self.missing_values = missing_values

        if n_components == None:
            n_components = 2
        else:
            self.n_components = n_components

        self.pca = PCA()


    def performPCA(self):
        self.pca.fit(self.data.X)



    def screePlot(self):

        self.performPCA()

        plt.figure(1, figsize=(4,3))
        plt.clf()
        plt.plot(self.pca.explained_variance_ratio_, linewidth=2)
        plt.axis('tight')
        plt.xlabel('Number of Components')
        plt.ylabel('Percentage of Variance')
        plt.show()


myPCA = testPCA(data, headers, missing_value)