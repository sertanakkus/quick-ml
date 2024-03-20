import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


import warnings

warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)


def SVC_(target_feature):
    # df = pd.read_csv("../saved_file.csv")
    df = pd.read_csv("temp.csv")

    # drop na columns
    df = df.dropna()

    # encoding
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
    le = LabelEncoder()
    if target_feature in non_numeric_columns:
        df[target_feature] = le.fit_transform(df[target_feature])

    df = pd.get_dummies(data=df, columns=non_numeric_columns.drop(target_feature))

    # split data
    X = df.drop(columns=target_feature).values
    y = df[target_feature].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    classifier = SVC()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred) * 100
    precision = metrics.precision_score(
        y_test, y_pred, pos_label="positive", average="micro"
    )
    recall = metrics.recall_score(y_test, y_pred, pos_label="positive", average="micro")
    f1_score = metrics.f1_score(y_test, y_pred, pos_label="positive", average="micro")

    score = round(score, 1)
    score = f"{score} %"

    return score


def SVR_(target_feature):
    # df = pd.read_csv("../saved_file.csv")
    df = pd.read_csv("temp.csv")

    # drop na columns
    df = df.dropna()

    # encoding
    non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
    le = LabelEncoder()
    if target_feature in non_numeric_columns:
        df[target_feature] = le.fit_transform(df[target_feature])

    df = pd.get_dummies(data=df, columns=non_numeric_columns.drop(target_feature))

    # split data
    X = df.drop(columns=target_feature).values
    y = df[target_feature].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    regressor = SVR()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    score = metrics.r2_score(y_test, y_pred) * 100
    score = round(score, 1)
    score = f"{score} %"

    return score
