from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console

console = Console()

# Sklearn provide multiples of datasets like load_iris, load_digits, load_boston, etc...
iris_dataset = load_iris()
# Here we split our dataset by roughly 80/20, 80% used to train our model and 20% used to ensure predictions are correct
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)
# iris_dataset is then converted to a pandas DataFrame for easier Manipulations
iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)

console.rule("Iris dataset")
console.print(iris_dataframe)

model = RandomForestClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

console.rule("Classifications result")
console.print(classification_report(y_test, predictions))

features_importance = pd.DataFrame(
    index=iris_dataframe.columns,
    data=model.feature_importances_,
    columns=["Feature Importance"],
).sort_values("Feature Importance")

console.rule("Features importance")
console.print(features_importance)

plot_confusion_matrix(model, x_test, y_test)
plt.show()
