import numpy as np
import pandas as pd
import sklearn

# ----------------------------------------------------------------------------------------------------------------------
# Generate some dummy data

data_income = np.random.randn(100)
data_bill_amount = np.random.randn(100)
print(
    f"generated a random array: \n\n {str(data_income)} \n\n of shape {data_income.shape}"
)

df = pd.DataFrame(data_income)
df.columns = ["income"]
df["bill_amount"] = data_bill_amount * 10000
df["income"] = df["income"].apply(lambda x: x * 1000)
print(f"turned our test data into an income dataframe...\n {df.head()}")

# ----------------------------------------------------------------------------------------------------------------------
# Get to the actual work.
print(
    "this time we'll be doing the same thing with sklearn, just to see what fklearn is trying to do."
)

# 1. First let's cap our data with numpy
df["income"] = np.clip(df["income"], a_min=None, a_max=500)

# 2. Now let's run our regression
from sklearn.linear_model import LinearRegression

X = np.array(df["income"].values).reshape(-1, 1)
reg = LinearRegression().fit(X, df["bill_amount"])
predictions = reg.predict(X)
df["predictions"] = predictions

# 3. finally lets cap the data as well
df["predictions"] = np.clip(df["predictions"], a_min=None, a_max=200)

#
print(f" the returned dataframe now contains our capped prediction:\n {df.head(5)}")
