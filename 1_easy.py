import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Generate some dummy data

data_income = np.random.randn(100)
data_bill_amount = np.random.randn(100)
print(f"generated a random array: \n\n {str(data_income)} \n\n of shape {data_income.shape}")

df = pd.DataFrame(data_income)
df.columns=["income"]
df["bill_amount"] = data_bill_amount*10000
df["income"] = df["income"].apply(lambda x:x*1000)
print(f"turned our test data into an income dataframe...\n {df.head()}")

# ----------------------------------------------------------------------------------------------------------------------
# Get to the actual work.

from fklearn.training.regression import linear_regression_learner
from fklearn.training.transformation import capper, floorer, prediction_ranger

# initialize several learner functions
# 1. one function to cap the input data to ignore outliers.
# 2. then a usual regression
# 3. third again we'd min/max the output of the regression

capper_fn = capper(columns_to_cap=["income"], precomputed_caps={"income": 500})
regression_fn = linear_regression_learner(features=["income"], target="bill_amount")
ranger_fn = prediction_ranger(prediction_min=0.0, prediction_max=200.0)

# apply two by currieing them together...
from fklearn.training.pipeline import build_pipeline
learner = build_pipeline(capper_fn,regression_fn, ranger_fn)
p, df, log = learner(df)

print(f" the returned dataframe now contains our capped prediction:\n {df.head(5)}")