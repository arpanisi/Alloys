from .regression_models import *
from .regression_utils import *
from .plot_utils import plot_limits
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

classical_regression_models = [linear_reg, ridge_reg, lasso_reg, elastic_net_reg,
                               svr_reg, nu_svr_reg, linear_svr_reg, decision_tree_reg,
                               knn_reg, kernel_ridge_reg,
                               mlp_reg, passive_aggressive_reg, ransac_reg, huber_reg, tweedie_reg,
                               poisson_reg, theil_sen_reg, tf_regression_model]

ensemble_regression_models = [random_forest_reg, gradient_boosting_reg,
                              ada_boost_reg, bagging_reg, voting_reg, extra_trees_reg,
                              hist_gradient_boosting_reg,
                              ]

probabilistic_regression_models = [gaussian_process_reg, tf_bnn_regression_model,
                                   tf_bnn_regression_vi, tf_prob_regression_model]

ensemble_regression_names = ["Random Forest Regression",
                             "Gradient Boosting Regression", "AdaBoost Regression",
                             "Bagging Regression", "Voting Regression", "Extra Trees Regression",
                             "Histogram-based GBR", ]

classical_regression_names = ["Linear Regression", "Ridge Regression", "Lasso Regression",
                              "Elastic Net Regression", "SVR", "NuSVR", "Linear SVR",
                              "Decision Tree Regression", "K-Nearest Neighbors Regression",
                              "Kernel Ridge Regression",
                              "MLP Regression", "Passive Aggressive Regression", "RANSAC Regression",
                              "Huber Regression", "Tweedie Regression", "Poisson Regression",
                              "Theil-Sen Regression", "Deep Neural Network Regression", ]

probabilistic_regression_names = ["Gaussian Process Regression", "BNN-Variational Regression",
                                  "BNN-Flipout Regression", "PNN-StudentT Regression"]


def run_regular_regression_models(
        X, y, prop_name: str, num_folds: int = 5, save_path: str = None, model_save_path: str = None,
):
    """
    Evaluate multiple regression models with K-Fold cross-validation and
    return a summary of metrics.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target values.
        prop_name (str): Name of the property being predicted (used in filename).
        num_folds (int): Number of CV folds.
        save_path (str or None): Optional path to save CSV summary.
        model_save_path(str or None): Optional path to save model weights.

    Returns:
        pd.DataFrame: Summary table of metrics per model.
    """
    print(f"\n[INFO] Running CV for property: {prop_name} with {num_folds} folds")
    print(f"[INFO] Total models: {len(classical_regression_names)}")

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Save y_max for rescaling
    y_max = y.max()
    y_scaled = y / y_max

    # Storage for results
    results = regression_metrics(classical_regression_models)

    # Generating Input Shape for Tensorflow model
    input_shape = X.shape[1:]

    # Iterate through each fold
    for fold_num, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f"\n[INFO] Fold {fold_num}/{num_folds}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_scaled.iloc[train_index], y_scaled.iloc[test_index]

        iterator = zip(classical_regression_names, classical_regression_models)
        for model_name, model in iterator:
            print(f"[DEBUG] Training model: {model_name}")
            if model_name == "Deep Neural Network Regression":
                model = model(input_shape)
                model.fit(X_train, y_train, epochs=4000)
            else:
                model.fit(X_train, y_train)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Rescale predictions back
            y_pred_train_rescaled = y_pred_train * y_max
            y_pred_test_rescaled = y_pred_test * y_max
            y_train_rescaled = y_train * y_max
            y_test_rescaled = y_test * y_max

            if model_name == "Deep Neural Network Regression":
                model_filename = f"{model_save_path}/{model_name.replace(' ', '_')}_{prop_name}_{fold_num}.h5"
                model.save(model_filename)
            else:
                model_filename = f"{model_save_path}/{model_name.replace(' ', '_')}_{prop_name}_{fold_num}.joblib"
                joblib.dump(model, model_filename)

            # Collect metrics
            reg_metrics = compute_regression_metrics(y_train_rescaled, y_pred_train_rescaled,
                                                     y_test_rescaled, y_pred_test_rescaled)
            # update results
            for key, value in reg_metrics.items():
                results[model_name][key].append(value)

    print("\n[INFO] Cross-validation complete. Summarizing results...")

    # Summarize: mean ± std across folds
    summarize_cv_results(results, prop_name, save_path=save_path)
    print(f"[INFO] Summary complete for {prop_name}.")


def run_ensemble_regression_models(
        X, y, prop_name: str, num_folds: int = 5, save_path: str = None,
        fig_dir_path: str = None, model_save_path: str = None,
):
    """
        Evaluate multiple regression models with K-Fold cross-validation and
        return a summary of metrics.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target values.
            prop_name (str): Name of the property being predicted (used in filename).
            num_folds (int): Number of CV folds.
            save_path (str or None): Optional path to save CSV summary.
            fig_dir_path (str or None): Optional path to save figures.
            model_save_path (str or None): Optional path to save models.

        """
    print(f"\n[INFO] Running CV for property: {prop_name} with {num_folds} folds")
    print(f"[INFO] Total models: {len(ensemble_regression_models)}")

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Save y_max for rescaling
    y_max = y.max()
    y_scaled = y / y_max

    # Storage for results
    results = regression_metrics(ensemble_regression_names, prob_model=True)

    # Iterate through each fold
    for fold_num, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f"\n[INFO] Fold {fold_num}/{num_folds}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_scaled.iloc[train_index], y_scaled.iloc[test_index]

        iterator = zip(ensemble_regression_names, ensemble_regression_models)
        for model_name, model in iterator:
            print(f"[DEBUG] Training model: {model_name}")

            model.fit(X_train, y_train)

            # Ensemble mean and std
            mean_tr, std_tr = ensemble_mean_std(model, X_train)
            mean_te, std_te = ensemble_mean_std(model, X_test)

            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Rescale predictions back
            y_pred_train_rescaled = y_pred_train * y_max
            y_pred_test_rescaled = y_pred_test * y_max
            y_train_rescaled = y_train * y_max
            y_test_rescaled = y_test * y_max

            # Save model
            model_filename = f"{model_save_path}/{model_name.replace(' ', '_')}_{prop_name}_{fold_num}.joblib"
            joblib.dump(model, model_filename)

            # uncertainty-like metrics (only if std is available)
            picp_tr, mpiw_tr, avgstd_tr, lower_tr, upper_tr = (
                interval_metrics(y_train.values, mean_tr, std_tr, alpha=1.96))
            picp_te, mpiw_te, avgstd_te, lower_te, upper_te = (
                interval_metrics(y_test.values, mean_te, std_te, alpha=1.96))

            # Metrics (MAE, MSE rescaled, R² unaffected by scaling)
            # Collect metrics
            reg_metrics = compute_regression_metrics(y_train_rescaled, y_pred_train_rescaled,
                                                     y_test_rescaled, y_pred_test_rescaled)
            # update results
            for key, value in reg_metrics.items():
                results[model_name][key].append(value)

            # Collect Uncertainty-aware metrics
            results[model_name]["PICP_train"].append(picp_tr)
            results[model_name]["PICP_test"].append(picp_te)
            results[model_name]["MPIW_train"].append(mpiw_tr)
            results[model_name]["MPIW_test"].append(mpiw_te)
            results[model_name]["STD_train_mean"].append(avgstd_tr)
            results[model_name]["STD_test_mean"].append(avgstd_te)

            if std_te is not None:
                plot_limits(y_test, mean_te, lower_te, upper_te, model_name=model_name,
                            prop_name=prop_name, save_path=fig_dir_path, fold_num=fold_num)

    print("\n[INFO] Cross-validation complete. Summarizing results...")

    # Summarize: mean ± std across folds
    summarize_cv_results(results, prop_name, save_path=save_path)
    print(f"[INFO] Summary complete for {prop_name}.")


def run_probabilistic_regression_models(
        X, y, prop_name: str, num_folds: int = 5, save_path: str = None,
        fig_dir_path: str = None, model_save_path: str = None,
):
    """
        Evaluate multiple regression models with K-Fold cross-validation and
        return a summary of metrics.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target values.
            prop_name (str): Name of the property being predicted (used in filename).
            num_folds (int): Number of CV folds.
            save_path (str or None): Optional path to save CSV summary.
            fig_dir_path (str or None): Optional path to save figures.
            model_save_path (str or None): Optional path to save models.
        """
    print(f"\n[INFO] Running CV for property: {prop_name} with {num_folds} folds")
    print(f"[INFO] Total models: {len(probabilistic_regression_models)}")

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Save y_max for rescaling
    y_max = y.max()
    y_scaled = y / y_max

    # Generating Input Shape for Tensorflow model
    input_shape = X.shape[1:]

    # Storage for results
    results = regression_metrics(probabilistic_regression_names, prob_model=True)

    # Iterate through each fold
    for fold_num, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f"\n[INFO] Fold {fold_num}/{num_folds}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_scaled.iloc[train_index], y_scaled.iloc[test_index]

        # Generating Input Size for BNN models
        input_size = len(X_train)

        iterator = zip(probabilistic_regression_names, probabilistic_regression_models)
        for model_name, model in iterator:
            print(f"[DEBUG] Training model: {model_name}")

            if model_name == "BNN-Variational Regression" or model_name == "BNN-Flipout Regression":
                model = model(input_data_size=input_size, input_shape=input_shape)
                model.fit(X_train, y_train, epochs=4000)
            elif model_name == "PNN-StudentT Regression":
                model = model(input_shape=input_shape)
                model.fit(X_train, y_train, epochs=4000)
            else:
                model.fit(X_train, y_train)

            # Ensemble mean and std
            mean_tr, std_tr = ensemble_mean_std(model, X_train, model_name)
            mean_te, std_te = ensemble_mean_std(model, X_test, model_name)

            # uncertainty-like metrics (only if std is available)
            picp_tr, mpiw_tr, avgstd_tr, lower_tr, upper_tr = (
                interval_metrics(y_train.values, mean_tr, std_tr, alpha=1.96))
            picp_te, mpiw_te, avgstd_te, lower_te, upper_te = (
                interval_metrics(y_test.values, mean_te, std_te, alpha=1.96))

            if model_name == "Gaussian Process Regression":
                model_filename = f"{model_save_path}/{model_name.replace(' ', '_')}_{prop_name}_{fold_num}.joblib"
                joblib.dump(model, model_filename)
            else:
                model_filename = f"{model_save_path}/{model_name.replace(' ', '_')}_{prop_name}_{fold_num}.h5"
                model.save_weights(model_filename)

            # Rescale predictions back
            y_pred_train_rescaled = mean_tr * y_max
            y_pred_test_rescaled = mean_te * y_max
            y_train_rescaled = y_train * y_max
            y_test_rescaled = y_test * y_max

            # Collect metrics
            reg_metrics = compute_regression_metrics(y_train_rescaled, y_pred_train_rescaled,
                                                     y_test_rescaled, y_pred_test_rescaled)
            # update results
            for key, value in reg_metrics.items():
                results[model_name][key].append(value)

            # Collect Uncertainty-aware metrics
            results[model_name]["PICP_train"].append(picp_tr)
            results[model_name]["PICP_test"].append(picp_te)
            results[model_name]["MPIW_train"].append(mpiw_tr)
            results[model_name]["MPIW_test"].append(mpiw_te)
            results[model_name]["STD_train_mean"].append(avgstd_tr)
            results[model_name]["STD_test_mean"].append(avgstd_te)

            if std_te is not None:
                plot_limits(y_test, mean_te, lower_te, upper_te, model_name=model_name,
                            prop_name=prop_name, save_path=fig_dir_path, fold_num=fold_num)

    print("\n[INFO] Cross-validation complete. Summarizing results...")

    # Summarize: mean ± std across folds
    summarize_cv_results(results, prop_name, save_path=save_path)
    print(f"[INFO] Summary complete for {prop_name}.")


def ensemble_mean_std(model, X: pd.DataFrame,
                      model_name: str = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns (mean_preds, std_preds) across base estimators where it makes sense.
    Supported: RandomForest, ExtraTrees, Bagging, VotingRegressor.
    For unsupported models, returns (point_preds, None).
    """

    # Bagging-like with accessible estimators_
    if isinstance(model, (RandomForestRegressor, ExtraTreesRegressor,
                          BaggingRegressor, VotingRegressor,
                          AdaBoostRegressor)):
        preds = np.stack([est.predict(X) for est in model.estimators_], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    if isinstance(model, GradientBoostingRegressor):
        preds = np.stack([est[0].predict(X) for est in model.estimators_], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    if (model_name == "BNN-Variational Regression" or model_name == "BNN-Flipout Regression"
            or model_name == "PNN-StudentT Regression"):
        preds = model(X.values)
        return preds.mean().numpy().flatten(), preds.stddev().numpy().flatten()

    if isinstance(model, GaussianProcessRegressor):
        y_mean, y_std = model.predict(X.values, return_std=True)
        return y_mean, y_std

    # Fallback: no estimator-level spread available (e.g., GradientBoosting, AdaBoost, HistGBR)
    # We can at least return point predictions; std=None signals “no uncertainty proxy”.
    return model.predict(X), None


def interval_metrics(y_true: np.ndarray, mean: np.ndarray, std: Optional[np.ndarray], alpha: float = 1.96):
    """
    Compute PICP and MPIW using mean ± alpha*std. If std is None, returns NaNs.
    """
    if std is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan  # PICP, MPIW, avg_std

    lower = mean - alpha * std
    upper = mean + alpha * std
    inside = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(inside))
    mpiw = float(np.mean(upper - lower))
    avg_std = float(np.mean(std))
    return picp, mpiw, avg_std, lower, upper
