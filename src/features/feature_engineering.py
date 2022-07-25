#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/data/feature_selection.py

"""

from __future__ import absolute_import, division, print_function

import pandas as pd


def booleans_to_int(df:pd.DataFrame) -> pd.DataFrame:
    """
    """
    df[df.select_dtypes('boolean').columns] = df.select_dtypes('boolean').astype(int)
    return df
    

def one_hot_encoding(df:pd.DataFrame) -> pd.DataFrame:
    """
    """
    df.loc[df['telekomHybridUploadSpeed'] == 10, 'telekomHybridUploadSpeed'] = 1
    df.loc[df['telekomHybridUploadSpeed'] != 10, 'telekomHybridUploadSpeed'] = 0

    df['energyEfficiencyClass'] = df['energyEfficiencyClass'].replace('NO_INFORMATION', 'C_OR_LOWER')
    df['energyEfficiencyClass'] = df['energyEfficiencyClass'].replace(np.nan, 'C_OR_LOWER')
    df.loc[df['energyEfficiencyClass'] == 'A_PLUS', 'energyEfficiencyClass'] = 'A'
    df.loc[df['energyEfficiencyClass'].str.contains('F|E|G|H|D|C', na=False), 'energyEfficiencyClass'] = 'C_OR_LOWER'
    df_energyEfficiencyClass = pd.get_dummies(df['energyEfficiencyClass'], prefix='energyEfficiencyClass')

    df['telekomTvOffer'] = df['telekomTvOffer'].replace('NONE', np.nan)
    df_telekomTvOffer = pd.get_dummies(df['telekomTvOffer'], prefix='telekomTvOffer', dummy_na=True)

    df['typeOfFlat'] = df['typeOfFlat'].fillna('other')
    df.loc[df['typeOfFlat'].str.contains('ground_floor|apartment|roof_storey|raised_ground_floor|other|half_basement', na=False), 'typeOfFlat'] = 'non_luxury_type'
    df_typeOfFlat = pd.get_dummies(df['typeOfFlat'], prefix='typeOfFlat')

    df.loc[df['interiorQuality'] == 'simple', 'interiorQuality'] = 'normal'
    df.loc[df['interiorQuality'] == 'sophisticated', 'interiorQuality'] = 'not_luxury'
    df_interiorQuality = pd.get_dummies(df['interiorQuality'], prefix='interiorQuality', dummy_na=True)

    df['petsAllowed'] = df['petsAllowed'].fillna('no')
    df.loc[df['petsAllowed'].str.contains('negotiable|yes'), 'petsAllowed'] = 'yes'
    df_petsAllowed = pd.get_dummies(df['petsAllowed'], prefix='petsAllowed')

    df.loc[df['numberOfFloors'] >= 5, 'numberOfFloors'] = 'more_than_5'
    df.loc[df['numberOfFloors'] == 4, 'numberOfFloors'] = '4'
    df.loc[df['numberOfFloors'] == 3, 'numberOfFloors'] = '3'
    df.loc[df['numberOfFloors'] == 2, 'numberOfFloors'] = '2'
    df.loc[df['numberOfFloors'] == 1, 'numberOfFloors'] = '1'
    df.loc[df['numberOfFloors'] == 0, 'numberOfFloors'] = '0'
    df_numberOfFloors = pd.get_dummies(df['numberOfFloors'], prefix='numberOfFloors', dummy_na=True)

    df.loc[df['lastRefurbish'] < 1950, 'lastRefurbish_cat'] = 'very old'
    df.loc[(df['lastRefurbish'] > 1950) & (df['lastRefurbish'] < 2000), 'lastRefurbish_cat'] = 'old'
    df.loc[(df['lastRefurbish'] > 2000) & (df['lastRefurbish'] < 2015), 'lastRefurbish_cat'] = 'new'
    df.loc[(df['lastRefurbish'] > 2015), 'lastRefurbish_cat'] = 'very new'
    df_lastRefurbish = pd.get_dummies(df['lastRefurbish_cat'], prefix='lastRefurbish_cat', dummy_na=True)

    df.loc[df['telekomUploadSpeed'] >= 40, 'telekomUploadSpeed_cat'] = 'fast'
    df.loc[df['telekomUploadSpeed'] < 40, 'telekomUploadSpeed_cat'] = 'slow'
    df_telekomUploadSpeed = pd.get_dummies(df['telekomUploadSpeed_cat'], prefix='telekomUploadSpeed_cat', dummy_na=True)

    return pd.concat([df_telekomTvOffer,
                    df_typeOfFlat, 
                    df_interiorQuality, 
                    df_petsAllowed, 
                    df_numberOfFloors, 
                    df_lastRefurbish, 
                    df_telekomUploadSpeed], axis=1)
