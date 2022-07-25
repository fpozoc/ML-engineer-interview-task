#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/data/preprocessing.py

"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/data/nlp.py

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
    return pd.read_csv(data_path, lineterminator='\n')


def create_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Create an index column.
    
    Args:
        df: A pandas DataFrame.
    
    Returns:
        A pandas DataFrame with an index column.
    """
    df['idx'] = range(1, len(df) + 1)
    df.insert(0, 'id', [f"{idx}.{totalRent} - {street} {houseNumber} {floor} ({town} - {city}, {state})" 
                for idx, state, city, town, street, houseNumber, floor, totalRent in zip(
                    df['idx'], df['regionLevel1'], df['regionLevel2'], df['regionLevel3'], df['street'], df['houseNumber'], df['floor'], df['totalRent'])])
    df = df.drop('idx', axis=1)
    return df


def add_features(df:pd.DataFrame, features_path:str) -> pd.DataFrame:
    """
    """
    df_features = load_features(features_path)
    features = df_features['feature']
    return df[['id']+list(features)]


def remove_non_target_rows(df:pd.DataFrame, target:str) -> pd.DataFrame:
    """
    """
    return df.loc[~df[target].isnull()]