#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/model/model_selection.py

Model training, model selection and model evaluation.

This file can also be imported as a module and contains the following classes:
    * Splitter - Data Splitter class.
    * Classifier - Inherits train/test/split functionality from Splitter class.
    * ModelSelection
"""

from __future__ import absolute_import, division, print_function

import os, pickle

import numpy as np
import pandas as pd
import random

from loguru import logger
from sklearn.model_selection import (
    KFold,
    cross_validate,
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    make_scorer,
    precision_score,
    recall_score,
    classification_report,
    classification_report,
)
from sklearn.metrics import *

from xgboost import XGBClassifier, XGBRegressor

__author__ = "Fernando Pozo"
__copyright__ = "Copyright 2022"
__license__ = "GNU General Public License"
__version__ = "0.0.1"
__maintainer__ = "Fernando Pozo"
__email__ = "fpozoc@gmx.com"
__status__ = "Development"


class Splitter:
    def __init__(
        self,
        df: pd.DataFrame,
        features_col: list,
        target_col: str,
        random_state: int,
        test_size: float = 0.25,
    ):
        """Data Splitter class

        Args:
            df (pd.DataFrame): DataFrame with the data.
            features_col (list): List of features columns.
            target_col (str): Target column.
            random_state (int): Random state.
            test_size (float, optional): Test size. Defaults to 0.25.
        """
        self.df = df
        self.target_col = target_col
        self.random_state = random_state
        (
            self.train_features,
            self.test_features,
            self.train_target,
            self.test_target,
        ) = train_test_split(
            df[features_col],
            df[target_col],
            test_size=test_size,
            random_state=random_state,
        )
        self.training_set = self.df.iloc[self.train_features.index]
        self.test_set = self.df.iloc[self.test_features.index]


class Classifier(Splitter):
    def __init__(
        self,
        model: object,
        df: list,
        features_col: list,
        target_col: str,
        model_type: str,
        random_state: int = 123,
        test_size: float = 0.25,
        preprocessing: object = None,
    ):
        """Inherits train/test/split functionality from Splitter class.

        Args:
            model (object): sklearn instance model.
            df (list): pandas DataFrame with training set.
            features_col (list): feature names to use as independent variables.
            target_col (str): target name to use as dependent variable.
            model_type (str): model type.
            random_state (int, optional): Seed. Defaults to 123.
            test_size (float, optional): Size to split the test. Defaults to 0.25.
            preprocessing (object, optional): Preprocessing step to be added to the model pipeline. Defaults to None.
        """
        super().__init__(df, features_col, target_col, random_state, test_size)
        self.features_col = features_col
        self.target_col = target_col
        self.model_type = model_type
        self.preprocessing = preprocessing
        self.model = model

        self.fit()

    def fit(self):
        self.pipeline = Pipeline(
            [("preprocessing", self.preprocessing), ("model", self.model)]
        )
        self.pipeline.fit(self.train_features, self.train_target)
        self.predictions = self.pipeline.predict(self.test_features)
        if self.model_type == "classification":
            self.probabilities = self.pipeline.predict_proba(self.test_features)[:, 1]

    @property
    def evaluate(self):
        scores = self._metrics()
        df_scores = pd.DataFrame(scores, index=["metric"]).T
        return df_scores

    @property
    def classification_report(self):
        return classification_report(self.test_target, self.predictions)

    @property
    def confusion_matrix(self):
        labels = ["TN", "FP", "FN", "TP"]
        cm = list(
            zip(labels, confusion_matrix(self.test_target, self.predictions).ravel())
        )
        df_cm = pd.DataFrame(cm).set_index(0).T
        return df_cm

    @property
    def cross_validate(self, n_splits: int = 5) -> dict:
        df = self._init_training_set()
        scoring_functions = self._scoring_functions()
        cv = cross_validate(
            self.model,
            df[self.features_col],
            df[self.target_col],
            cv=n_splits,
            scoring=scoring_functions,
            return_train_score=True,
        )
        results = pd.DataFrame(cv)
        results = list(
            zip(results.columns, results.mean().round(4), results.std().round(4))
        )
        df_results = pd.DataFrame(results, columns=["metric", "mean", "std"]).set_index(
            "metric"
        )
        return df_results

    def make_prediction(self, samples: list, probability: bool = False):
        if probability:
            return self.pipeline.predict_proba(samples)[:, 1]
        else:
            return self.pipeline.predict(samples)

    def save_model(self, outdir: str, name: str = "custom_model.pkl"):
        with open(os.path.join(outdir, name), "wb") as model_filepath:
            pickle.dump(self.model, model_filepath)

    def _init_training_set(self):
        train = pd.concat([self.train_features, self.train_target], axis=1)
        test = pd.concat([self.test_features, self.test_target], axis=1)
        return pd.concat([train, test], axis=0).reset_index(drop=True)

    def _metrics(
        self,
    ) -> dict:
        if self.model_type == "classification":
            metrics = {
                "Accuracy": accuracy_score(self.test_target, self.predictions),
                "AUC": roc_auc_score(self.test_target, self.probabilities),
                "Average Precision Score": average_precision_score(
                    self.test_target, self.probabilities
                ),
                "Balanced Accuracy": balanced_accuracy_score(
                    self.test_target, self.predictions
                ),
                "F1 Score": f1_score(self.test_target, self.predictions),
                "Log Loss": -1 * (log_loss(self.test_target, self.probabilities)),
                "MCC": matthews_corrcoef(self.test_target, self.predictions),
                "Precision": precision_score(self.test_target, self.predictions),
                "Recall": recall_score(self.test_target, self.predictions),
            }
        elif self.model_type == "regression":
            metrics = {
                "R2": r2_score(self.test_target, self.predictions),
                "Mean Absolute Error": mean_absolute_error(
                    self.test_target, self.predictions
                ),
                "Mean Squared Error": mean_squared_error(
                    self.test_target, self.predictions
                ),
            }
        return metrics

    def _scoring_functions(self):
        if self.model_type == "classification":
            scores = {
                "Accuracy": "accuracy",
                "AUC": "roc_auc",
                "Average Precision Score": "average_precision",
                "Balanced Accuracy": "balanced_accuracy",
                "F1 Score": "f1",
                "Log Loss": "neg_log_loss",
                "MCC": make_scorer(matthews_corrcoef),
                "Precision": "precision",
                "Recall": "recall",
                "Sensivity": make_scorer(recall_score),
                "Specificity": make_scorer(recall_score, pos_label=0),
            }
        elif self.model_type == "regression":
            scores = {
                "R2": "r2",
                "Mean Absolute Error": "neg_mean_absolute_error",
                "Mean Squared Error": "neg_mean_squared_error",
            }
        return scores


class ModelSelection(Splitter):
    def __init__(
        self,
        df: pd.DataFrame,
        features_col: list,
        target_col: str,
        model_type: str,
        random_state: int = 123,
        n_outer_splits: int = 5,
        n_inner_splits: int = 10,
        n_jobs: int = 20,
        save: bool = False,
        filepath: str = None,
    ):
        """Inherits train/test/split functionality from Splitter class.

        Args:
            df (pd.DataFrame): pandas DataFrame with training set.
            features_col (list): feature names to use as independent variables.
            target_col (str): target name to use as dependent variable.
            random_state (int, optional): Seed. Defaults to 123.
            model_type (str): Type of model to use. 'classification' or 'regression'.
            n_outer_splits (int, optional): Number of splits to use in the Outer Cross Validation. Defaults to 5.
            n_inner_splits (int, optional): Number of splits to use in the Outer Cross Validation. Defaults to 10.
            n_jobs (int, optional): Number of processors. Defaults to 20.
            save (bool, optional): Save selected model option. Defaults to False.
            filepath (str, optional): Save selected model file path. Defaults to None.
        """
        super().__init__(df, features_col, target_col, random_state)
        self.n_train = self.train_features.shape[0]
        self.n_test = self.test_features.shape[0]
        self.random_state = random_state
        self.model_type = model_type
        self.n_outer_splits = n_outer_splits
        self.n_inner_splits = n_inner_splits
        self.n_jobs = n_jobs

    def get_best_model(self, outdir: str = None, selection_metric: str = None) -> str:
        """Getting best model from Nested Cross Validation.

        Args:
            selection_metric (str, optional): Metric to refit the GridSearchCV. Defaults to 'MCC'.

        Returns:
            [str]: Selected model with the best metric performance.
        """
        if outdir:
            # utils.create_dir(outdir)
            self._create_file_log(outdir)

        if selection_metric is None:
            if self.model_type == "classification":
                selection_metric = "MCC"
            elif self.model_type == "regression":
                selection_metric = "Mean Absolute Error"

        gridscv = dict()
        for model, config in self._model_configurations().items():
            if config["enable"]:
                gcv = GridSearchCV(
                    estimator=config[self.model_type],
                    param_grid=config["grid1"],
                    scoring=self._scoring_functions(),
                    n_jobs=self.n_jobs,
                    cv=self._inner_cv(),
                    verbose=0,
                    refit=f"{selection_metric}",
                    return_train_score=True,
                )
                gridscv[model] = gcv

        models = dict()
        for model, _ in sorted(gridscv.items()):
            logger.info(50 * "-", "\n")
            logger.info(f"Algorithm: {model}")
            logger.info(f"\tInner loop:")

            outer_scores = []
            models[model] = dict()
            for i, (train_idx, valid_idx) in enumerate(
                self._outer_cv().split(self.train_features, self.train_target)
            ):
                logger.info(f"\t({i+1})")

                gcvhp = gridscv[model]
                gcvhp.fit(
                    self.train_features.values[train_idx],
                    self.train_target.values[train_idx],
                )

                gcvhp_results = self._get_cv_results(gcvhp.cv_results_)
                models[model][i + 1] = gcvhp_results
                best_model = gcvhp.best_estimator_.fit(
                    self.train_features.values[train_idx],
                    self.train_target.values[train_idx],
                )
                scores = self._evaluate(
                    best_model,
                    self.train_features.values[valid_idx],
                    self.train_target.values[valid_idx],
                )
                outer_scores.append(scores[f"{selection_metric}"])

                logger.info(f"\tBest params: {gcvhp_results['params']}")
                logger.info(
                    f"\tTrain {selection_metric}: {gcvhp_results[f'mean_train_{selection_metric}']}"
                )
                logger.info(
                    f"\tTest {selection_metric}: {gcvhp_results[f'mean_test_{selection_metric}']}"
                )

            outer_score = (
                f"{round(np.mean(outer_scores),3)} +/- {round(np.std(outer_scores),3)}"
            )
            models[model][f"val_mean_{selection_metric}"] = outer_score

            logger.info(
                f"\tAvg. {selection_metric} (on validation folds): {outer_score}"
            )

        self.results = pd.DataFrame(models).T.reset_index()
        best_model = pd.DataFrame(models).T.sort_values(
            by=f"val_mean_{selection_metric}", ascending=False
        )[:1]
        best_rounds = []
        for i in range(1, 6):
            best_rounds.append(
                (i, best_model[i].values[0][f"mean_test_{selection_metric}"])
            )

        best_params = best_model[
            sorted(best_rounds, key=lambda x: x[1], reverse=True)[0][0]
        ].values[0]["params"]
        selected_model = (
            self._model_configurations()[f"{best_model.index[0]}"][self.model_type]
            .set_params(**best_params)
            .fit(self.train_features, self.train_target)
        )

        self.model = selected_model
        self.model.fit(self.train_features, self.train_target)

        if outdir:
            self.save_model(outdir)
            self.save_results(outdir)

        return self.model

    def save_model(self, outdir: str) -> None:
        with open(os.path.join(outdir, "selected_model.pkl"), "wb") as model_filepath:
            pickle.dump(self.model, model_filepath)

    def save_results(self, outdir: str) -> None:
        self.results.to_csv(
            os.path.join(outdir, "model_selection.tsv.gz"),
            index=None,
            compression="gzip",
            sep="\t",
        )

    def _create_file_log(self, outdir: str) -> str:
        logger.add(os.path.join(outdir, "model_selection_{time}.log"))
        init_msg = f'Nested CV ({self.n_outer_splits} outer folds \
         - {self.n_inner_splits} innter folds) Model Selection will be \
         performed with {self.n_train} training instances ({self.n_test} test instances) using \
         a Random State with seed=12{self.random_state} and {self.n_jobs} processors.\n{10*"-"}\n\n'
        logger.info(init_msg)

    def _outer_cv(self, shuffle: bool = False) -> StratifiedKFold:
        if self.model_type == "classification":
            cv = StratifiedKFold(n_splits=self.n_outer_splits, shuffle=shuffle)
        elif self.model_type == "regression":
            cv = KFold(n_splits=self.n_outer_splits, shuffle=shuffle)
        return cv

    def _inner_cv(self, shuffle: bool = True) -> StratifiedKFold:
        if self.model_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.n_inner_splits,
                shuffle=shuffle,
                random_state=self.random_state,
            )
        elif self.model_type == "regression":
            cv = KFold(
                n_splits=self.n_outer_splits,
                shuffle=shuffle,
                random_state=self.random_state,
            )
        return cv

    def _evaluate(self, model: object, features: list, target: list) -> dict:
        """Evaluating model with the following metrics.

        Args:
            model (object): sklearn pre-fitted model.
            features (list): test features to evaluate.
            target (list): test target.

        Returns:
            dict: metrics of the evaluation
        """
        predictions = model.predict(features)
        if self.model_type == "classification":
            probs = model.predict_proba(features)[:, 1]
            scores = {
                "Accuracy": accuracy_score(target, predictions),
                "AUC": roc_auc_score(target, probs),
                "Average Precision Score": average_precision_score(target, probs),
                "Balanced Accuracy": balanced_accuracy_score(target, predictions),
                "F1 Score": f1_score(target, predictions),
                "Log Loss": -1 * (log_loss(target, probs)),
                "MCC": matthews_corrcoef(target, predictions),
                "Precision": precision_score(target, predictions),
                "Recall": recall_score(target, predictions),
            }
        elif self.model_type == "regression":
            scores = {
                "R2": r2_score(target, predictions),
                "Mean Absolute Error": mean_absolute_error(target, predictions),
                "Mean Squared Error": mean_squared_error(target, predictions),
            }
        return scores

    def _scoring_functions(self):
        if self.model_type == "classification":
            scores = {
                "Accuracy": "accuracy",
                "AUC": "roc_auc",
                "Average Precision Score": "average_precision",
                "Balanced Accuracy": "balanced_accuracy",
                "F1 Score": "f1",
                "Log Loss": "neg_log_loss",
                "MCC": make_scorer(matthews_corrcoef),
                "Precision": "precision",
                "Recall": "recall",
                "Sensivity": make_scorer(recall_score),
                "Specificity": make_scorer(recall_score, pos_label=0),
            }
        elif self.model_type == "regression":
            scores = {
                "R2": "r2",
                "Mean Absolute Error": "neg_mean_absolute_error",
                "Mean Squared Error": "neg_mean_squared_error",
            }
        return scores

    def _model_configurations(self):
        grid = {
            "AdaBoost": {
                "enable": False,
                "shortname": "AB",
                "classification": AdaBoostClassifier(
                    n_estimators=400, random_state=self.random_state
                ),
                "regression": "None",
                "grid1": [{"learning_rate": [0.001, 0.01, 0.1]}],
            },
            "Decision Tree": {
                "enable": True,
                "shortname": "DT",
                "classification": DecisionTreeClassifier(
                    random_state=self.random_state
                ),
                "regression": DecisionTreeRegressor(random_state=self.random_state),
                "grid1": [
                    {
                        "min_samples_leaf": list(range(5, 15)),
                    }
                ],
            },
            "Extremely randomized trees": {
                "enable": False,
                "shortname": "ET",
                "classification": ExtraTreesClassifier(
                    n_estimators=400, random_state=self.random_state
                ),
                "regression": "None",
                "grid1": [
                    {
                        "min_samples_leaf": list(range(4, 8)),
                        "max_features": list(range(1, 8)),
                    }
                ],
            },
            "K-Nearest Neighbors": {
                "enable": False,
                "shortname": "KNN",
                "classification": Pipeline(
                    [
                        ("std", StandardScaler()),
                        (
                            "knn",
                            KNeighborsClassifier(algorithm="ball_tree", leaf_size=50),
                        ),
                    ]
                ),
                "regression": "None",
                "grid1": [{"knn__n_neighbors": list(range(1, 10)), "knn__p": [1, 2]}],
            },
            "Logistic Regression": {
                "enable": False,
                "shortname": "LR",
                "classification": Pipeline(
                    [
                        ("std", StandardScaler()),
                        (
                            "lg",
                            LogisticRegression(
                                multi_class="multinomial",
                                solver="newton-cg",
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                ),
                "regression": "None",
                "grid1": [
                    {
                        "lg__penalty": ["l1", "l2"],
                        "lg__C": np.power(10.0, np.arange(-4, 4)),
                    }
                ],
            },
            "Random Forest": {
                "enable": True,
                "shortname": "RF",
                "classification": RandomForestClassifier(
                    n_estimators=400, random_state=self.random_state, n_jobs=10
                ),
                "regression": RandomForestRegressor(
                    n_estimators=400, random_state=self.random_state, n_jobs=-1
                ),
                "grid1": [
                    {
                        "min_samples_leaf": list(range(2, 7)),
                    }
                ],
            },
            "Gradient Boosting Machine": {
                "shortname": "GBM",
                "enable": True,
                "classification": GradientBoostingClassifier(
                    n_estimators=400, random_state=self.random_state
                ),
                "regression": GradientBoostingRegressor(
                    n_estimators=400, random_state=self.random_state
                ),
                "grid1": [
                    {
                        "min_samples_leaf": list(range(2, 7)),
                        "learning_rate": [0.01, 0.1, 0.2],
                    }
                ],
            },
            "Support Vector Machine": {
                "enable": False,
                "shortname": "SVM",
                "classification": Pipeline(
                    [
                        ("std", StandardScaler()),
                        ("svm", SVC(probability=True, random_state=self.random_state)),
                    ]
                ),
                "regression": "None",
                "grid1": [
                    {
                        "svm__kernel": ["rbf"],
                        "svm__C": np.power(10.0, np.arange(-4, 4)),
                        "svm__gamma": np.power(10.0, np.arange(-5, 0)),
                    },
                    {
                        "svm__kernel": ["linear"],
                        "svm__C": np.power(10.0, np.arange(-4, 4)),
                    },
                ],
            },
            "XGBoost": {
                "shortname": "XGB",
                "enable": True,
                "classification": XGBClassifier(
                    scale_pos_weight=4, seed=self.random_state
                ),
                "regression": XGBRegressor(n_estimators=400, seed=self.random_state),
                "grid1": [
                    {
                        # 'min_child_weight': [1, 5, 10],
                        # 'gamma': [0.5, 1, 1.5, 2, 5],
                        # 'subsample': [0.6, 0.8, 1.0],
                        # 'colsample_bytree': [0.6, 0.8, 1.0],
                        "max_depth": list(range(2, 7))
                    }
                ],
            },
        }
        return grid

    def _get_cv_results(self, cv_results_: dict) -> pd.DataFrame:
        """Processing cross-validation results from:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

        Args:
            cv_results_ (dict): dictionary with GridSearchCV results.
            metric_sort (str, optional): metric to sort the pandas DataFrame generated.

        Returns:
            pd.DataFrame: pandas DataFrame that summarizes the results.
        """
        if self.model_type == "classification":
            metric_sort = "MCC"
        elif self.model_type == "regression":
            metric_sort = "Mean Absolute Error"
        df = pd.DataFrame(cv_results_)
        cv_results_dict = (
            df[["params"] + [col for col in df if col.startswith("mean_t")]]
            .sort_values(by=f"mean_test_{metric_sort}", ascending=False)
            .to_dict("records")[0]
        )
        return cv_results_dict
