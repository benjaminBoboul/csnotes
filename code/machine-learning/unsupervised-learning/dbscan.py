from sklearn.cluster import DBSCAN
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

boston_dataset = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    boston_dataset["data"], boston_dataset["target"], random_state=0
)

pca = PCA(n_components=3, whiten=False, random_state=0)
pca_x_train = pca.fit_transform(x_train)

boston_dataframe = DataFrame(pca_x_train)

print(boston_dataframe)

# eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
# min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
# leaf_size: Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. 
# 			 The optimal value depends on the nature of the problem.
# n_jobs: The number of parallel jobs to run. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
dbscan = DBSCAN(eps=25, min_samples=5, leaf_size=30, n_jobs=4)

y_pred = dbscan.fit_predict(pca_x_train)
# boston_dataframe["class"] = y_pred

# print(boston_dataframe)

fig = plt.figure()
sns.color_palette()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    xs=boston_dataframe[0], ys=boston_dataframe[1], zs=boston_dataframe[2], c=y_pred
)
plt.title("DBSCAN")
plt.show()
