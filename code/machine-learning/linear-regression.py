from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import median_absolute_error
from sklearn.pipeline import make_pipeline
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import seaborn as sns

boston_dataset = load_boston()

x_train, x_test, y_train, y_test = train_test_split(
    boston_dataset["data"], boston_dataset["target"], random_state=0
)

model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)

predictions = model.predict(x_test)

predictions_dataframe = df(
    data=[predictions, y_test], index=["prediction", "expected value"]
).T

print(predictions_dataframe)
print("Median abs. error: ", median_absolute_error(y_test, predictions))

# Finally, plot the predicted prices with the expected ones
sns.set_theme(style="white", palette="pastel")
sns.lineplot(data=predictions_dataframe)
plt.show()
