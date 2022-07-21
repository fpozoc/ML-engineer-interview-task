#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/model/train.py

Usage: python -m src.model.train --features config/features.yaml --pretrained
__
--help              |-h     Display documentation.
--custom            |-c     Train with a customized model.
--features          |-f     Features selected description yaml filepath.
--model_selection   |-m     Performs the model selection protocol.
--pretrained        |-p     Train with a previously trained model.
"""

from __future__ import absolute_import, division, print_function

import argparse, os, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor

from model_selection import Classifier, ModelSelection
from ..utils import utils

__author__ = "Fernando Pozo"
__copyright__ = "Copyright 2022"
__license__ = "GNU General Public License"
__version__ = "0.0.1"
__maintainer__ = "Fernando Pozo"
__email__ = "fpozoc@gmx.com"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser(
        description='Command-line arguments parser', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d", "--dataset",  
        help="Training set.")
    parser.add_argument(
        "-c", "--custom",  action='store_true', default=False, 
        help="Training and saving a customized model.")
    parser.add_argument('-f', '--features', default='config/features.yaml', 
        help="Features selected description filepath.")
    parser.add_argument(
        "-m", "--model_selection",  action='store_true', default=False, 
        help="Perform a nested cv model selection, training and saving the best model.")
    parser.add_argument(
        "-p", "--pretrained",  action='store_true', default=False, 
        help="Train with a previously trained model.")
    parser.add_argument(
        "-s", "--seed",  default=123,
        help="Perform a nested cv model selection, training and saving the best model.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    df = pd.read_csv(Path('../../data/training_set.tsv.gz').resolve(), sep='\t', compression='gzip')

    if args.model_selection:
        ms = ModelSelection(df[df.columns[1:]].head(200), 
                            features_col=df.columns[2:],
                            target_col='totalRent',
                            model_type='regression',
                            random_state=args.seed)

        model = ms.get_best_model(outdir=Path('../../models').resolve())

    elif args.custom:
        custom_model = RandomForestRegressor(
            n_estimators=100, 
            min_samples_leaf = 3,
            random_state=args.seed)

        model = Classifier(
            model=RandomForestRegressor(n_estimators=1000, min_samples_leaf=1, random_state=SEED),
            df=df[df.columns[1:]],
            features_col=df.columns[2:],
            target_col='totalRent',
            model_type='regression',
            )
        model.save_model(outdir=Path('../../models').resolve())
                

    elif args.pretrained:
        pretrained_model = pickle.load(open(os.path.join(Path('../../models').resolve(), 'selected_model.pkl'), 'rb'))
        model = Classifier(
            model=pretrained_model,
            df=df[df.columns[1:]],
            features_col=df.columns[2:],
            target_col='totalRent',
            model_type='regression',
            )


if __name__ == "__main__":
    main()