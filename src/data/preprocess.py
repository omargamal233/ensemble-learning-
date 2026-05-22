import pandas as pd
import numpy as np

from .load import load_data
from ..features.engineer import (
    create_cabin_feature,
    create_family_features,
    create_ticket_feature,
    create_title_feature,
    encode_categorical,
    select_features,
)


def detect_outliers(df: pd.DataFrame, n: int, features: list) -> list:
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col].dropna(), 25)
        Q3 = np.percentile(df[col].dropna(), 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = pd.Series(outlier_indices).value_counts()
    return list(outlier_indices[outlier_indices > n].index)


def impute_age(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["Age"] = dataset.groupby(["Pclass", "Sex", "SibSp", "Parch"])["Age"].transform(
        lambda group: group.fillna(group.median())
    )
    dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
    return dataset


def fill_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
    dataset = impute_age(dataset)
    return dataset


def prepare_dataset(train: pd.DataFrame, test: pd.DataFrame, drop_outliers: bool = True):
    if drop_outliers:
        drop_indices = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
        train = train.drop(index=drop_indices).reset_index(drop=True)

    train_len = len(train)
    dataset = pd.concat([train, test], sort=False).reset_index(drop=True)
    dataset = fill_missing_values(dataset)
    dataset = create_title_feature(dataset)
    dataset = create_cabin_feature(dataset)
    dataset = create_family_features(dataset)
    dataset = create_ticket_feature(dataset)
    dataset = encode_categorical(dataset)

    passenger_ids = dataset.loc[train_len:, "PassengerId"].reset_index(drop=True)
    dataset = select_features(dataset)

    train_processed = dataset.iloc[:train_len].copy()
    test_processed = dataset.iloc[train_len:].copy()
    train_processed["Survived"] = train_processed["Survived"].astype(int)

    X_train = train_processed.drop(columns=["Survived"])
    y_train = train_processed["Survived"]
    X_test = test_processed.drop(columns=["Survived"])

    return X_train, y_train, X_test, passenger_ids


def load_and_prepare(data_dir=None, drop_outliers=True):
    train, test = load_data(data_dir)
    return prepare_dataset(train, test, drop_outliers=drop_outliers)
