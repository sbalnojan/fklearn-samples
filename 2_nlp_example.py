import sklearn
import fklearn
import gluonnlp
import mxnet

import gluonnlp as nlp
train_dataset, test_dataset = [nlp.data.IMDB(root='data/imdb', segment=segment)
                                for segment in ('train', 'test')]
data = train_dataset._data + test_dataset._data
print(train_dataset[0][0])
print(f"score: {train_dataset[0][1]}")

import pandas as pd
df = pd.DataFrame(data, columns=["text","score"])
print(f"turning the data into dataframe format ....\n {df.head(5)} ")

#df = df.sample(10000)
# ----------------------------------------------------------------------------------------------------------------------
# Define a new function that could be applied to any kind of text data


from toolz import curry
import lightgbm
from fklearn.types import LearnerReturnType
from sklearn.feature_extraction.text import TfidfVectorizer

@curry
def fit_tfidf(train_set: pd.DataFrame,
              text_column: str,
              target_column: str,
              max_features: int) -> LearnerReturnType:
    """ A curried tfidf fitter to be used before classification with one of the other classifiers...
    """
    vec = TfidfVectorizer(max_features=max_features)
    vec.fit(train_set[text_column])
    def p(transformed_df: pd.DataFrame) -> pd.DataFrame:
        transformed_data = vec.transform(train_set[text_column])
        transformed_df = pd.DataFrame(transformed_data.toarray())
        transformed_df[target_column] = train_set[target_column].values
        transformed_df[target_column].apply(lambda x: str(x))
        return transformed_df
    log = {'tfidfVectorizer': {'Nope':None}}

    return p, p(train_set), log

# ----------------------------------------------------------------------------------------------------------------------
# Doing the training

# Some evaluation splitting
from fklearn.validation.evaluators import fbeta_score_evaluator
from sklearn.model_selection import train_test_split
train_df, holdout_df = train_test_split(df)

transform_fn = fit_tfidf(text_column="text", target_column="score", max_features=5000)

_, transformed_df,_ = transform_fn(df)
print("transformed data... running training: ...")

# Pre transformations

p, df, log = transform_fn(train_df)

from fklearn.training.classification import lgbm_classification_learner
predict_fn, df, log =lgbm_classification_learner(df, features=transformed_df.columns[transformed_df.columns != "score"]
,target="score", encode_extra_cols=False, extra_params={"verbose":1, "objective": "multiclass", "num_class":11,
                                                        "learning_rate": 0.05,
                                                        "max_depth": 5,
                                                        "max_bin":255,
                                                        "num_leaves": 31,
                                                        "feature_fraction": 0.5,
                                                        "lambda_l1": .2})

# ----------------------------------------------------------------------------------------------------------------------
# Doing the predicting on the holdout set

p, df, log = transform_fn(holdout_df)
predictions = predict_fn(df)

# ----------------------------------------------------------------------------------------------------------------------
# Transforming the predictions, run evaluation

import numpy as np

holdout_preds = predictions[predictions.columns[-11:]].values
holdout_class_preds = np.argmax(holdout_preds, axis=1)
holdout_df["prediction"] = holdout_class_preds

from sklearn.metrics import fbeta_score
f_beta = fbeta_score(holdout_df["score"], holdout_df["prediction"], average='macro', beta=0.5)
print(f"F1 Score: {f_beta}")