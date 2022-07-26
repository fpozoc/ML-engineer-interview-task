#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/data/preprocessing.py

"""

from __future__ import absolute_import, division, print_function

import pandas as pd
from ..utils.utils import *


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file.

    error with this file reported here https://stackoverflow.com/questions/33998740/error-in-reading-a-csv-file-in-pandascparsererror-error-tokenizing-data-c-err

    Args:
        data_path: Path to the CSV file.

    Returns:
        A pandas DataFrame.
    """
    return pd.read_csv(data_path, lineterminator="\n")


def create_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Create an index column.

    Args:
        df: A pandas DataFrame.

    Returns:
        A pandas DataFrame with an index column.
    """
    df["idx"] = range(1, len(df) + 1)
    df.insert(
        0,
        "id",
        [
            f"{idx}.{totalRent} - {street} {houseNumber} {floor} ({town} - {city}, {state})"
            for idx, state, city, town, street, houseNumber, floor, totalRent in zip(
                df["idx"],
                df["regionLevel1"],
                df["regionLevel2"],
                df["regionLevel3"],
                df["street"],
                df["houseNumber"],
                df["floor"],
                df["totalRent"],
            )
        ],
    )
    df = df.drop("idx", axis=1)
    return df


def add_features(df: pd.DataFrame, features_path: str) -> pd.DataFrame:
    """Add features to the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to be converted.
        features_path (str): Path to the features yaml file.

    Returns:
        pd.DataFrame: Converted dataframe.
    """
    df_features = load_features(features_path)
    features = df_features["feature"]
    return df[["id"] + list(features)]


def remove_non_target_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Remove rows with non-target values.

    Args:
        df (pd.DataFrame): Dataframe to be converted.

    Returns:
        pd.DataFrame: Converted dataframe.
    """
    return df.loc[~df[target].isnull()]


def booleans_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean to integer conversion.

    Args:
        df (pd.DataFrame): Dataframe to be converted.

    Returns:
        pd.DataFrame: Converted dataframe.
    """
    df[df.select_dtypes("boolean").columns] = df.select_dtypes("boolean").astype(int)
    return df


def one_hot_encoding(df: pd.DataFrame, features_yaml_path: str) -> pd.DataFrame:
    """One hot encoding of categorical variables.

    Args:
        df (pd.DataFrame): Dataframe to be encoded.
        features_yaml_path (str): Path to the features yaml file.

    Returns:
        pd.DataFrame: Encoded dataframe.
    """
    df.loc[df["telekomHybridUploadSpeed"] == 10, "telekomHybridUploadSpeed"] = 1
    df.loc[df["telekomHybridUploadSpeed"] != 10, "telekomHybridUploadSpeed"] = 0

    df["energyEfficiencyClass"] = df["energyEfficiencyClass"].replace(
        "NO_INFORMATION", "C_OR_LOWER"
    )
    df["energyEfficiencyClass"] = df["energyEfficiencyClass"].replace(
        np.nan, "C_OR_LOWER"
    )
    df.loc[df["energyEfficiencyClass"] == "A_PLUS", "energyEfficiencyClass"] = "A"
    df.loc[
        df["energyEfficiencyClass"].str.contains("F|E|G|H|D|C", na=False),
        "energyEfficiencyClass",
    ] = "C_OR_LOWER"
    df_energyEfficiencyClass = pd.get_dummies(
        df["energyEfficiencyClass"], prefix="energyEfficiencyClass"
    )

    df["telekomTvOffer"] = df["telekomTvOffer"].replace("NONE", np.nan)
    df_telekomTvOffer = pd.get_dummies(
        df["telekomTvOffer"], prefix="telekomTvOffer", dummy_na=True
    )

    df["typeOfFlat"] = df["typeOfFlat"].fillna("other")
    df.loc[
        df["typeOfFlat"].str.contains(
            "ground_floor|apartment|roof_storey|raised_ground_floor|other|half_basement",
            na=False,
        ),
        "typeOfFlat",
    ] = "non_luxury_type"
    df_typeOfFlat = pd.get_dummies(df["typeOfFlat"], prefix="typeOfFlat")

    df.loc[df["interiorQuality"] == "simple", "interiorQuality"] = "normal"
    df.loc[df["interiorQuality"] == "sophisticated", "interiorQuality"] = "not_luxury"
    df_interiorQuality = pd.get_dummies(
        df["interiorQuality"], prefix="interiorQuality", dummy_na=True
    )

    df["petsAllowed"] = df["petsAllowed"].fillna("no")
    df.loc[df["petsAllowed"].str.contains("negotiable|yes"), "petsAllowed"] = "yes"
    df_petsAllowed = pd.get_dummies(df["petsAllowed"], prefix="petsAllowed")

    df.loc[df["numberOfFloors"] >= 5, "numberOfFloors"] = "more_than_5"
    df.loc[df["numberOfFloors"] == 4, "numberOfFloors"] = "4"
    df.loc[df["numberOfFloors"] == 3, "numberOfFloors"] = "3"
    df.loc[df["numberOfFloors"] == 2, "numberOfFloors"] = "2"
    df.loc[df["numberOfFloors"] == 1, "numberOfFloors"] = "1"
    df.loc[df["numberOfFloors"] == 0, "numberOfFloors"] = "0"
    df_numberOfFloors = pd.get_dummies(
        df["numberOfFloors"], prefix="numberOfFloors", dummy_na=True
    )

    df.loc[df["lastRefurbish"] < 1950, "lastRefurbish_cat"] = "very old"
    df.loc[
        (df["lastRefurbish"] > 1950) & (df["lastRefurbish"] < 2000), "lastRefurbish_cat"
    ] = "old"
    df.loc[
        (df["lastRefurbish"] > 2000) & (df["lastRefurbish"] < 2015), "lastRefurbish_cat"
    ] = "new"
    df.loc[(df["lastRefurbish"] > 2015), "lastRefurbish_cat"] = "very new"
    df_lastRefurbish = pd.get_dummies(
        df["lastRefurbish_cat"], prefix="lastRefurbish_cat", dummy_na=True
    )

    df.loc[df["telekomUploadSpeed"] >= 40, "telekomUploadSpeed_cat"] = "fast"
    df.loc[df["telekomUploadSpeed"] < 40, "telekomUploadSpeed_cat"] = "slow"
    df_telekomUploadSpeed = pd.get_dummies(
        df["telekomUploadSpeed_cat"], prefix="telekomUploadSpeed_cat", dummy_na=True
    )

    df_features = pd.DataFrame(parse_yaml(features_yaml_path))

    return pd.concat(
        [
            df[
                df_features[
                    df_features["training"].str.contains("^included$", na=False)
                ]["feature"].values
            ],
            df_energyEfficiencyClass,
            df_telekomTvOffer,
            df_typeOfFlat,
            df_interiorQuality,
            df_petsAllowed,
            df_numberOfFloors,
            df_lastRefurbish,
            df_telekomUploadSpeed,
        ],
        axis=1,
    )


def remove_missing_values(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Removes missing values.

    Args:
        df (pd.DataFrame): Dataframe to be cleaned.
        feature (str): Column to be cleaned.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    return df.loc[~df[col].isnull()]


def remove_outliers(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Removes outliers by list of features.

    Args:
        df (pd.DataFrame): Dataframe to be cleaned.
        features (list): List of features to be cleaned.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    for feature in features:
        df = remove_outliers_by_quantile(df, feature)
    return df
