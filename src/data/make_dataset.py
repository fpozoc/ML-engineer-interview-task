#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/data/make_dataset.py

This script creates a dataframe for the mode for task 1.

Usage:
    python -m src.data.make_dataset -d data/raw/immo_data.csv -o data/processed/training_set.v1.tsv.gz
    make make-dataset -d data/raw/immo_data.csv -o data/processed/training_set.v1.tsv.gz
"""

from __future__ import absolute_import, division, print_function

import argparse
import pandas as pd

from ..features.preprocessing import *


def main():
    parser = argparse.ArgumentParser(
        description='Command-line arguments parser', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d", "--dataset",  default='data/raw/immo_data.csv',   
        help="Training set.")
    parser.add_argument('-f', '--features', default='config/features.yaml', 
        help="Features selected description filepath.")
    parser.add_argument(
        "-o", "--output",  
        help="Training set.")
    args = parser.parse_args()

    df = create_model_df(args.dataset, args.features)
    df.to_csv(args.output, sep='\t', compression='gzip', index=False)


def create_model_df(raw_data_path:str, features_yaml_path:str)->pd.DataFrame:
    """Create a dataframe for the model.

    Args:
        raw_data_path (str): Path to the raw data.
        features_yaml_path (str): Path to the features yaml file.
    
    Returns:
        pd.DataFrame: Dataframe for the model.
    """
    df = load_data(raw_data_path)
    df = create_idx(df)
    df = add_features(df, features_yaml_path)
    df = remove_non_target_rows(df, 'totalRent')
    df = booleans_to_int(df)
    df = one_hot_encoding(df)
    df = remove_outliers(df, ['serviceCharge', 'livingSpace', 'noRooms', 'totalRent'])
    return df


if __name__ == "__main__":
    main()