#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/utils/utils.py

Helper Functions

Classes and functions:
    * create_dir: returns the path to directory recently created.
    * impute - returns a pandas DataFrame with features imputed.
    * merge_dataframes: returns a merged DataFrame on transcript identifier.
    * one_hot_encoding - returns a pandas DataFrame with features encoded.
    * reduce_mem_usage: returns a pandas pandas DataFrame with optimized memory usage.
    * reorder_cols: returns a pandas DataFrame with columns reordered by object and then float columns.
    * round_df_floats: returns pandas DataFrame with float features rounded.
    * parse_yaml: returns a pandas DataFrame with new features.
    * timer: returns info from time consumed in a coding block.
    * unity_ranger - returns a pandas DataFrame with features between 0 and 1. 
"""

from __future__ import absolute_import, division, print_function

from contextlib import contextmanager
import functools, gzip, os, yaml

import numpy as np
import pandas as pd
from loguru import logger 

__author__ = "Fernando Pozo"
__copyright__ = "Copyright 2022"
__license__ = "GNU General Public License"
__version__ = "0.0.1"
__maintainer__ = "Fernando Pozo"
__email__ = "fpozoc@gmx.com"
__status__ = "Development"


def create_dir(dirpath: str) -> str: 
    """mkdir -p python equivalent
    
    Arguments:
        dirpath {str} -- Path to create the new folder
    
    Returns:
        absolute_path {str} -- Absolute path of the new folder
    """    
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    absolute_path = os.path.abspath(dirpath)


def get_df_info(df:pd.DataFrame) -> str:
    """Getting pandas DataFrame info for logging.

    Args:
        df (pd.DataFrame): data set as input.

    Returns:
        str: Info message.
    """    
    ncols = df.shape[1]
    nrows = df.shape[0]
    colstr = ';'.join(df.columns)
    msg = f"{ncols} features ({colstr}) with {nrows} instances loaded."
    return msg


def impute(df:pd.DataFrame, features:list, n:int=None, itype:str='class', column:str=None, condition:str=None, percentile:float=None) -> pd.DataFrame:
    """Imputator

    It contains some functions to impute vectors in different ways.

    Args:
        df (pd.DataFrame): input data set.
        features (list): List of features to normalize.
        n (int, optional): Maximum number of score value. Defaults to None.
        itype (str, optional): Type of imputation to perform. Defaults to 'class'.
        column (str, optional): Column to select in condition. Defaults to None.
        condition(str, optional): Value to select in condition. Defaults to None.
        percentile(float, optional): Percentile to impute with. Defaults to None.

    Returns:
        pd.DataFrame: data set with imputed columns.
    """    
    for feature in features:
        if itype == 'class':
            df.loc[df[feature].isnull(), feature] = n
        elif itype == 'conditional':
            df.loc[df[column].str.contains(condition, na=False), feature] = n
        elif itype == 'percentile':
            df.loc[df[feature].isnull(), feature] = df[feature].quantile(percentile/100)  
        elif itype == 'same_as_norm':
            df.loc[df[feature].isnull(), feature] = df[feature.replace('norm_','')]         
    return df


def merge_dataframes(*args, on_type:str='transcript_id', how_type:str='left', pivot_on:int=0, nimpute:int=None) -> pd.DataFrame:
    """Pandas DataFrame merger Function.
    
    This function merges several DataFrame to create an unique database which contains same isoforms 
    as reference database. It uses pandas merge method.
    
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html


    Args:
        on_type (str, optional): Merge on feature selected. Defaults to 
    'transcript_id'.
        how_type (str, optional): Merge method. Defaults to 'left'.
        pivot_on (int, optional): Represent the first DataFrame to merge when 
    how_type='left'. Defaults to 0.
        nimpute (int, optional): If user wants to impute some feature, nimpute
    is the number to fill na. Defaults to None.

    Raises:
        ValueError: Merge function has to receive more than one DataFrame inside 
    list argument

    Returns:
        pd.DataFrame: pandas DataFrame mergefd
    """    
    args = list(args)
    if len(args) <= 1:
        raise ValueError(
            'Merge function has to receive more than one DataFrame inside list argument.')
    if how_type != 'left':
        pivot_on == None
    args.insert(0, args.pop(pivot_on))
    df = functools.reduce(lambda left, right: pd.merge(
        left, right, on=on_type, how=how_type),args).drop_duplicates(subset=on_type).reset_index(drop=True)
    if nimpute != None:
        df = df.fillna(nimpute)
    return df


def one_hot_encoding(df:pd.DataFrame, features:list) -> pd.DataFrame:
    """One Hot Encoder

    It encodes features selected as "one hot" mood and removing initial feature.
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    Args:
        df (pd.DataFrame): input data set.
        features (list): Feature list to perform the method

    Returns:
        pd.DataFrame: data set with encoded features.
    """    
    for feature in features:
        df[feature] = df[feature].astype(int)
        one_hot = pd.get_dummies(df[feature], prefix=feature)
        df = df.drop(feature, axis=1)
        df = df.join(one_hot)
    return df


def open_files(filepath:str) -> object:
    """Openning both compressed and non-compressed files.

    Args:
        filepath (str): File path to the file.

    Returns:
        str: open file object.
    """    
    if filepath.endswith('.gz'):
        open_file = gzip.open(filepath, 'rt')
    else: 
        open_file = open(filepath, 'r')
    return open_file


def parse_yaml(yaml_file:str) -> str:
    '''YAML parsing function

    This function parses a configuration file in yaml format (http://zetcode.com/python/yaml/).

    Parameters
    ----------
    yaml_file: str
        Config file path in yaml format.

    Returns
    -------
    config: dict
        Dictionary with configuration data structure.
    '''
    with open(yaml_file, 'r') as config:
        try:
            config = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            logger.info(exc)
    return config


def reduce_mem_usage(df:pd.DataFrame, verbose:bool=False, round_float:int=False) -> pd.DataFrame:
    """Memory reducer Function

    It reduces memory usage of pandas DataFrame. 
    Inspired in https://www.kaggle.com/artgor/artgor-utils

    Args:
        df (pd.DataFrame): input data set.
        verbose (bool, optional): Verbosity control. Defaults to False.
        round_float (int, optional): Round to 4 dec. Defaults to False.

    Returns:
        pd.DataFrame: data set with reduced memory usage.
    """
    start_mem_usg = df.memory_usage().sum() / 1024**2
    logger.info("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []
    for col in df.columns:
        if df[col].dtype != object:
            if verbose:
                logger.info(
                    "******************************\nColumn: {}\ndtype before: {}\n".format(col, df[col].dtype))
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            if not np.isfinite(df[col]).all():
                NAlist.append(col)
                df[col].fillna(mn-1, inplace=True)
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
                if round_float:
                    df[col] = df[col].round(round_float)
            if verbose:
                logger.info("dtype after: ", df[col].dtype)
    logger.info("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2
    logger.info("Memory usage is: ", mem_usg, " MB")
    logger.info("This is ", 100*mem_usg/start_mem_usg, "% of the initial size")
    return df, NAlist


def reorder_cols(df:pd.DataFrame) -> pd.DataFrame:
    """Reorder columns Function

    Ordering columns of DataFrame: Strings at the start of the df

    Args:
        df (pd.DataFrame): pandas DataFrame

    Returns:
        pd.DataFrame: pandas DataFrame
    """    
    return df[list(df.select_dtypes(include='object').columns)+list(df.select_dtypes(exclude='object').columns)]


def round_df_floats(df:pd.DataFrame, n:int=4) -> pd.DataFrame:
    """Rounder floats Function

    It rounds to 4 (default) all the floated columns of the DataFrame

    Args:
        df (pd.DataFrame): input data set.
        n (int, optional): round integer. Defaults to 4.

    Returns:
        pd.DataFrame: data set with floated columns rounded
    """    
    df[df.select_dtypes(include='float', exclude=None).columns] = df.select_dtypes(include='float', exclude=None).round(n)
    return df


@contextmanager
def timer(title:str):
    """
    https://docs.python.org/3/library/contextlib.html

    """
    import time
    t0 = time.time()
    yield
    logger.info("{} - done in {:.0f}m".format(title, round((time.time()-t0)/60), 2))


def unity_ranger(df:pd.DataFrame, features:list) -> pd.DataFrame:
    """Unity ranger Function

    It truncates to 1 values higher than 1 and to 0 values lower than 0.

    Args:
        df (pd.DataFrame): input data set.
        features (list): Feature list to perform the method

    Returns:
        pd.DataFrame: data set with corrected features.
    """    
    for feature in features:
        df.loc[df[feature] > 1, feature] = 1
        df.loc[df[feature] < 0, feature] = 0
    return df
