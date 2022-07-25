#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/data/make_dataset.py

"""

from __future__ import absolute_import, division, print_function

import pandas as pd

from ..features.feature_engineering import *
from ..features.feature_selection import *
from ..features.preprocessing import *

def create_model_df(raw_data_path:str, features_yaml_path:str)->pd.DataFrame:
    """
    """
    df = load_data(raw_data_path)
    df = create_idx(df)
    df = add_features(df, features_yaml_path)
    df = remove_non_target_rows(df, 'totalRent')
    return df