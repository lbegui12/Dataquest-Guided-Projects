import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 
sphist = pd.read_csv("sphist.csv")


# Convert "Date" to pandas date type
sphist["Date"] = pd.to_datetime(sphist["Date"])
sphist.sort_values(by="Date",ascending=False)

# Create new indicators
# average based
sphist["avg_last_5d"] = sphist["Close"].rolling(5).mean().shift(1)
sphist["avg_last_30d"] = sphist["Close"].rolling(30).mean().shift(1)
sphist["avg_last_365d"] = sphist["Close"].rolling(365).mean().shift(1)
sphist["avg_ratio_5_over_365"] = sphist["avg_last_5d"] / sphist["avg_last_365d"]
# std based indicators
sphist["std_last_5d"] = sphist["Close"].rolling(5).std().shift(1)
sphist["std_last_30d"] = sphist["Close"].rolling(30).std().shift(1)
sphist["std_last_365d"] = sphist["Close"].rolling(365).std().shift(1)
sphist["std_ratio_5_over_365"] = sphist["std_last_5d"] / sphist["std_last_365d"]

# Clean the dataset
sphist = sphist[ sphist["Date"] > datetime(1951, 1, 3)] # Before that date some indicators are useless
sphist = sphist.fillna(0)

# Split the dataset
train = sphist[sphist["Date"] < datetime(2013, 1, 1)] 
test = sphist[sphist["Date"] >= datetime(2013, 1, 1)] 


column_to_remove = ["Close", "High", "Low", "Open", "Volume", "Adj Close", "Date"]
features = list(set(train.columns.tolist())-set(column_to_remove))

lr = LinearRegression()
lr.fit(train[features], train["Close"])
predications = lr.predict(test[features])

mae = mean_absolute_error(test["Close"], predications)
mse = mean_squared_error(test["Close"], predications)

print("MAE = {} \t MSE = {}".format( mae, mse))

plt.scatter(test["Close"], predications)