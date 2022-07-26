#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/model/explainer.py

https://explainerdashboard.readthedocs.io/en/latest/

This file can also be imported as a module and contains the following classes:
    * Explainer
"""

from __future__ import absolute_import, division, print_function

import argparse, os, pickle, warnings

import pandas as pd
from explainerdashboard import RegressionExplainer, ExplainerDashboard

from .model_selection import Classifier
from ..utils.utils import *

__author__ = "Fernando Pozo"
__copyright__ = "Copyright 2022"
__license__ = "GNU General Public License"
__version__ = "0.0.1"
__maintainer__ = "Fernando Pozo"
__email__ = "fpozoc@gmx.com"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser(
        description="Command-line arguments parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--dataset", help="Training set.")
    parser.add_argument("-m", "--model", help="Model to explain.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    df = pd.read_csv(args.dataset, sep="\t", compression="gzip")

    my_model = pickle.load(open(args.model, 'rb'))
    model = Classifier(
        model=my_model,
        df=df[df.columns[1:]],
        features_col=df.columns[2:],
        target_col='totalRent',
        model_type='regression',
        )
    my_model.fit(model.train_features, model.train_target)

    explainer = RegressionExplainer(model, model.test_features, model.test_target)
    ExplainerDashboard(explainer).run()


if __name__ == "__main__":
    main()
