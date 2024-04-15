import logging
from math import sqrt
import pickle
import warnings
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from scipy.linalg import LinAlgWarning
from optuna.samplers import TPESampler

warnings.filterwarnings(action="ignore", category=LinAlgWarning)
warnings.filterwarnings(action="ignore")

logging.getLogger("optuna").setLevel(logging.ERROR)


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error**2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    loss = np.where(is_small_error, squared_loss, linear_loss)
    return np.mean(loss)


dataset_path = "data/parq/kd_dataset.parquet"
df = pd.read_parquet(dataset_path, engine="fastparquet")

feature_cols = ["GRAN_0_10", "GRAN_10_25", "GRAN_25_40", "GRAN_40_80"]
target_cols = ["М10", "М40"]

df = df[[*feature_cols, *target_cols]]
stats_df = pd.DataFrame()


def get_cat_boost_model(params):
    model_params = {
        "depth": params["depth"],
        "learning_rate": params["learning_rate"],
        "iterations": params["iterations"],
        "l2_leaf_reg": params["l2_leaf_reg"],
        "boosting_type": params["boosting_type"],
        "bootstrap_type": params["bootstrap_type"],
        "leaf_estimation_method": params["leaf_estimation_method"],
        "loss_function": params["loss_function"],
        "subsample": params["subsample"],
        "grow_policy": params["grow_policy"],
        "rsm": params["rsm"],
        "random_strength": params["random_strength"],
    }

    return CatBoostRegressor(verbose=False, **model_params, allow_writing_files=False)


def get_xgb_model(params):
    model_params = {
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "n_estimators": params["n_estimators"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "min_child_weight": params["min_child_weight"],
    }

    return XGBRegressor(**model_params)


def get_linear_regression(params):
    pol_params = {
        "degree": params["degree"],
        "interaction_only": params["interaction_only"],
        "include_bias": params["include_bias"],
    }

    reg_params = {
        "fit_intercept": params["fit_intercept"],
        "copy_X": params["copy_X"],
        "positive": params["positive"],
    }

    return Pipeline(
        [
            ("Pol", PolynomialFeatures(**pol_params)),
            ("Reg", LinearRegression(**reg_params)),
        ]
    )


def get_cat_boost_model_trial(trial: optuna.Trial):
    model_params = {
        "depth": trial.suggest_int("depth", 1, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "iterations": trial.suggest_int("iterations", 10, 150),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-5, 3, log=True),
        "boosting_type": trial.suggest_categorical(
            "boosting_type", ["Plain", "Ordered"]
        ),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bernoulli", "MVS"]
        ),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Gradient", "Newton", "Exact", "Simple"]
        ),
        "loss_function": trial.suggest_categorical(
            "loss_function", ["RMSE", "MAE", "MAPE"]
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        ),
        "rsm": trial.suggest_float("rsm", 0.5, 1),
        "random_strength": trial.suggest_float("random_strength", 0.5, 1),
    }

    return CatBoostRegressor(
        verbose=False,
        allow_writing_files=False,
        early_stopping_rounds=15,
        **model_params,
    )


def get_xgb_model_trial(trial: optuna.Trial):
    model_params = {
        "max_depth": trial.suggest_int("max_depth", 1, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 150),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 3, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "eta": trial.suggest_float("eta", 0.001, 0.1),
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear"]),
        "tree_method": trial.suggest_categorical(
            "tree_method", ["auto", "exact", "approx", "hist", "gpu_hist"]
        ),
        "grow_policy": trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        ),
    }

    return XGBRegressor(**model_params, early_stopping_rounds=30)


def get_linear_regression_trial(trial: optuna.Trial):
    pol_params = {
        "degree": trial.suggest_int("degree", 1, 3),
        "interaction_only": trial.suggest_categorical(
            "interaction_only", [True, False]
        ),
        "include_bias": trial.suggest_categorical("include_bias", [True, False]),
    }

    reg_params = {
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "copy_X": trial.suggest_categorical("copy_X", [True, False]),
        "positive": trial.suggest_categorical("positive", [True, False]),
    }

    return Pipeline(
        [
            ("Pol", PolynomialFeatures(**pol_params)),
            ("Reg", LinearRegression(**reg_params)),
        ]
    )


models_infos = [
    {"trials": 200, "type": "LinearRegression"},
    {"trials": 400, "type": "CatBoost"},
    {"trials": 400, "type": "XGB"},
]

x = df[feature_cols]
y = df[target_cols]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

for target in target_cols:
    y_train_target = y_train[target]
    y_val_target = y_val[target]

    for model_info in models_infos:
        model_type = model_info["type"]

        def optimize_model(trial: optuna.Trial):
            metrics = {"rmse": [], "huber_loss": [], "mae": []}

            for train_index, test_index in KFold(n_splits=5).split(
                x_train, y_train_target
            ):

                x_cv_train, x_cv_val = (
                    x_train.iloc[train_index],
                    x_train.iloc[test_index],
                )
                y_cv_train, y_cv_val = (
                    y_train_target.iloc[train_index],
                    y_train_target.iloc[test_index],
                )

                try:
                    if model_type == "CatBoost":
                        model = get_cat_boost_model_trial(trial)
                        model.fit(x_cv_train, y_cv_train, eval_set=(x_cv_val, y_cv_val))
                    elif model_type == "XGB":
                        model = get_xgb_model_trial(trial)
                        model.fit(
                            x_cv_train,
                            y_cv_train,
                            eval_set=[
                                (x_cv_train, y_cv_train),
                                (x_cv_val, y_cv_val),
                            ],
                            verbose=False,
                        )
                    else:
                        model = get_linear_regression_trial(trial)
                        model.fit(x_cv_train, y_cv_train)
                except BaseException as e:
                    raise optuna.TrialPruned()

                y_cv_pred = model.predict(x_cv_val)

                metrics["rmse"].append(sqrt(mean_squared_error(y_cv_val, y_cv_pred)))
                metrics["huber_loss"].append(huber_loss(y_cv_val, y_cv_pred))
                metrics["mae"].append(mean_absolute_error(y_cv_val, y_cv_pred))

            metric_values = np.array(list(metrics.values())).mean(axis=1)
            return 1 / np.mean(np.power(metric_values, -1))

        sampler = TPESampler()
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            optimize_model, n_trials=model_info["trials"], show_progress_bar=True
        )

        print(
            f"[{target}, {model_type}]: {study.best_value:.5f}\n{study.best_params}\n"
        )

        if model_type == "CatBoost":
            model = get_cat_boost_model(study.best_params)
            model.fit(x_train, y_train_target)
        elif model_type == "XGB":
            model = get_xgb_model(study.best_params)
            model.fit(
                x_train,
                y_train_target,
                eval_set=[
                    (x_train, y_train_target),
                    (x_val, y_val_target),
                ],
                verbose=False,
            )
        else:
            model = get_linear_regression(study.best_params)
            model.fit(x_train, y_train_target)

        y_val_pred = model.predict(x_val)
        y_pred = model.predict(x)

        y_target = y[target]

        stats_df = pd.concat(
            [
                stats_df,
                pd.DataFrame(
                    [
                        {
                            "MODEL": model_type,
                            "TARGET": target,
                            "TYPE": "test",
                            "PARAMETER": "MAE",
                            "VALUE": mean_absolute_error(y_val_target, y_val_pred),
                        },
                        {
                            "MODEL": model_type,
                            "TARGET": target,
                            "TYPE": "test",
                            "PARAMETER": "MSE",
                            "VALUE": mean_squared_error(y_val_target, y_val_pred),
                        },
                        {
                            "MODEL": model_type,
                            "TARGET": target,
                            "TYPE": "test",
                            "PARAMETER": "R2",
                            "VALUE": r2_score(y_val_target, y_val_pred),
                        },
                        {
                            "MODEL": model_type,
                            "TARGET": target,
                            "TYPE": "all",
                            "PARAMETER": "MAE",
                            "VALUE": mean_absolute_error(y_target, y_pred),
                        },
                        {
                            "MODEL": model_type,
                            "TARGET": target,
                            "TYPE": "all",
                            "PARAMETER": "MSE",
                            "VALUE": mean_squared_error(y_target, y_pred),
                        },
                        {
                            "MODEL": model_type,
                            "TARGET": target,
                            "TYPE": "all",
                            "PARAMETER": "R2",
                            "VALUE": r2_score(y_target, y_pred),
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

        df[f"{target}_{model_type}_PRED"] = y_pred

        with open(f"models/{target}_{model_type}_model.pkl", "wb") as file:
            pickle.dump(model, file)

with pd.ExcelWriter("data/xlsx/results/results.xlsx") as writer:
    df.to_excel(writer, sheet_name="data")
    stats_df.to_excel(writer, sheet_name="stats", index=False)