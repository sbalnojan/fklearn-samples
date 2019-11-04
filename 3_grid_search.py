import numpy as np
import pandas as pd
# ----------------------------------------------------------------------------------------------------------------------
# Generate some dummy data

data_income = np.random.randn(100)
data_bill_amount = np.random.randn(100)
print(
    f"generated a random array: \n\n {str(data_income)} \n\n of shape {data_income.shape}"
)

df = pd.DataFrame(data_income)
df.columns = ["income"]
df["target"] = data_bill_amount * 10000
df["income"] = df["income"].apply(lambda x: x * 1000)
print(f"turned our test data into an income dataframe...\n {df.head()}")

# ----------------------------------------------------------------------------------------------------------------------
# Generate some dummy data

from toolz import curry
from fklearn.tuning.parameter_tuners import grid_search_cv
from fklearn.training.regression import linear_regression_learner
from fklearn.validation.splitters import k_fold_splitter
from fklearn.validation.evaluators import mse_evaluator

space = {
    'intercept': lambda: [0, 1],
}


@curry
def param_train_fn(space, train_set):
    return linear_regression_learner(
        features=["income"],
        target="target",
        params={"fit_intercept": space["intercept"]})(train_set)


split_fn = k_fold_splitter(n_splits=2)
tuning_log = grid_search_cv(space,
                            df,
                            param_train_fn=param_train_fn,
                            split_fn=split_fn,
                            eval_fn=mse_evaluator)

for run in tuning_log:
    print(f"mse score in this run per fold: {run['validator_log']}")
    print(f"with parameters: {run['iter_space']}")
