import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

import warnings

warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)


def decision_tree_regressor(target_feature):
    # df = pd.read_csv("../saved_file.csv")
    df = pd.read_csv("temp.csv")

    # drop na columns
    df = df.dropna()

    # encoding
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
    le = LabelEncoder()
    if target_feature in non_numeric_columns:
        df[target_feature] = le.fit_transform(df[target_feature])

    df = pd.get_dummies(
        data=df, columns=non_numeric_columns.drop(target_feature, errors="ignore")
    )

    # split data
    X = df.drop(columns=target_feature).values
    y = df[target_feature].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    score = r2_score(y_test, y_pred) * 100
    score = round(score, 1)
    score = f"{score} %"

    return score
