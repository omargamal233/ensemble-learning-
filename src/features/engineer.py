import numpy as np
import pandas as pd


def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    title = parts[1].split(".")[0].strip()
    return title


def normalize_title(title: str) -> str:
    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
        "Sir", "Jonkheer", "Dona"
    ]
    if title in rare_titles:
        return "Rare"
    if title in ["Mlle", "Ms"]:
        return "Miss"
    if title == "Mme":
        return "Mrs"
    if title in ["Miss", "Mrs", "Master", "Mr"]:
        return title
    return "Rare"


def get_ticket_prefix(ticket: str) -> str:
    if pd.isna(ticket):
        return "X"
    cleaned = ticket.replace(".", "").replace("/", "").strip().upper()
    parts = cleaned.split()
    if len(parts) == 0:
        return "X"
    prefix = parts[0]
    if prefix.isdigit():
        return "X"
    return prefix


def create_cabin_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["Cabin"] = dataset["Cabin"].fillna("X")
    dataset["Cabin"] = dataset["Cabin"].apply(lambda cabin: cabin[0] if cabin != "X" else "X")
    return dataset


def create_title_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["Title"] = dataset["Name"].apply(extract_title).apply(normalize_title)
    return dataset


def create_family_features(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
    dataset["Single"] = (dataset["Fsize"] == 1).astype(int)
    dataset["SmallF"] = (dataset["Fsize"] == 2).astype(int)
    dataset["MedF"] = ((dataset["Fsize"] >= 3) & (dataset["Fsize"] <= 4)).astype(int)
    dataset["LargeF"] = (dataset["Fsize"] >= 5).astype(int)
    return dataset


def create_ticket_feature(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["Ticket"] = dataset["Ticket"].apply(get_ticket_prefix)
    return dataset


def encode_categorical(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1}).astype(int)
    dataset = pd.get_dummies(dataset, columns=["Title", "Embarked", "Cabin", "Ticket", "Pclass"], prefix=["Title", "Em", "Cabin", "T", "Pc"], dummy_na=False)
    return dataset


def select_features(dataset: pd.DataFrame) -> pd.DataFrame:
    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    dataset = dataset.drop(columns=[col for col in drop_columns if col in dataset.columns])
    return dataset
