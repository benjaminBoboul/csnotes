from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

console = Console()

iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
console.rule("Iris dataset")
console.print(iris_dataframe)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

console.rule("Classification results")
console.print(classification_report(y_test, predictions))

plot_confusion_matrix(model, x_test, y_test)
plt.show()

plt.figure(figsize=(12, 8), dpi=200)
plot_tree(model, feature_names=iris_dataframe.columns, filled=True)
plt.show()

features_importance = pd.DataFrame(index=iris_dataframe.columns, data=model.feature_importances_, columns=["Feature Importance"]).sort_values("Feature Importance")
console.rule("Features importance")
console.print(features_importance)
