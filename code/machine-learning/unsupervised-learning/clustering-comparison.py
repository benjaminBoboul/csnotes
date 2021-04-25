from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Common arguments
min_samples = 5

# Sklearn provide multiples of datasets like load_iris, load_digits, load_boston, etc...
dataset = load_iris()
# Here we split our dataset by roughly 80/20, 80% used to train our model and 20% used to ensure predictions are correct
x_train, x_test, y_train, y_test = train_test_split(
    dataset["data"], dataset["target"], random_state=0
)

train_dataframe = pd.DataFrame(x_train, columns=dataset.feature_names)
print(train_dataframe)

pca = PCA(n_components=2, whiten=False, random_state=0)
pca_x_train = pca.fit_transform(x_train)

pca_dataframe = pd.DataFrame(pca_x_train)
print(pca_dataframe)

sns.scatterplot(data=train_dataframe)
plt.show()