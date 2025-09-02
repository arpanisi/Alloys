from .regression_models import *
from .plot_utils import plot_limits
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
               "Theil-Sen Regression", "Deep Neural Network Regression",]

probabilistic_regression_names = ["Gaussian Process Regression", "BNN-Variational Regression",
               "BNN-Flipout Regression", "PNN-StudentT Regression"]


def run_regular_regression_models(
    X, y, prop_name: str, num_folds: int = 5, save_path: str = None
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

    Returns:
        pd.DataFrame: Summary table of metrics per model.
    """
    print(f"\n[INFO] Running CV for property: {prop_name} with {num_folds} folds")
    print(f"[INFO] Total models: {len(classical_regression_names)}")

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Storage for results
    results = {name: {"R2_train": [], "R2_test": [],
                      "MAE_train": [], "MAE_test": [],
                      "MSE_train": [], "MSE_test": []}
               for name in classical_regression_names}

    # Generating Input Shape for Tensorflow model
    input_shape = X.shape[1:]

    # Iterate through each fold
    for fold_num, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f"\n[INFO] Fold {fold_num}/{num_folds}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

            # Collect metrics
            results[model_name]["R2_train"].append(r2_score(y_train, y_pred_train))
            results[model_name]["R2_test"].append(r2_score(y_test, y_pred_test))
            results[model_name]["MAE_train"].append(mean_absolute_error(y_train, y_pred_train))
            results[model_name]["MAE_test"].append(mean_absolute_error(y_test, y_pred_test))
            results[model_name]["MSE_train"].append(mean_squared_error(y_train, y_pred_train))
            results[model_name]["MSE_test"].append(mean_squared_error(y_test, y_pred_test))

    print("\n[INFO] Cross-validation complete. Summarizing results...")

    # Summarize: mean ± std across folds
    summary_rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Model")

    # Save CSV if requested
    if save_path:
        filename = f"{save_path}/{prop_name}_cv_results.csv"
        summary_df.to_csv(filename)
        print(f"[INFO] Results saved to {filename}")

    print(f"[INFO] Summary complete for {prop_name}.")


def run_ensemble_regression_models(
    X, y, prop_name: str, num_folds: int = 5, save_path: str = None,
    fig_dir_path: str = None
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

        """
    print(f"\n[INFO] Running CV for property: {prop_name} with {num_folds} folds")
    print(f"[INFO] Total models: {len(ensemble_regression_models)}")

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Storage for results
    results = {name: {"R2_train": [], "R2_test": [],
                      "MAE_train": [], "MAE_test": [],
                      "MSE_train": [], "MSE_test": [],
                      "PICP_train": [], "PICP_test": [],
                      "MPIW_train": [], "MPIW_test": [],
                      "STD_train_mean": [], "STD_test_mean": [],
                      }
               for name in ensemble_regression_names}

    # Iterate through each fold
    for fold_num, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f"\n[INFO] Fold {fold_num}/{num_folds}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

            # uncertainty-like metrics (only if std is available)
            picp_tr, mpiw_tr, avgstd_tr, lower_tr, upper_tr = (
                interval_metrics(y_train.values, mean_tr, std_tr, alpha=1.96))
            picp_te, mpiw_te, avgstd_te, lower_te, upper_te = (
                interval_metrics(y_test.values, mean_te, std_te, alpha=1.96))

            # Collect metrics
            results[model_name]["R2_train"].append(r2_score(y_train, y_pred_train))
            results[model_name]["R2_test"].append(r2_score(y_test, y_pred_test))
            results[model_name]["MAE_train"].append(mean_absolute_error(y_train, y_pred_train))
            results[model_name]["MAE_test"].append(mean_absolute_error(y_test, y_pred_test))
            results[model_name]["MSE_train"].append(mean_squared_error(y_train, y_pred_train))
            results[model_name]["MSE_test"].append(mean_squared_error(y_test, y_pred_test))

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
    summary_rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Model")

    # Save CSV if requested
    if save_path:
        filename = f"{save_path}/{prop_name}_cv_results.csv"
        summary_df.to_csv(filename)
        print(f"[INFO] Results saved to {filename}")

    print(f"[INFO] Summary complete for {prop_name}.")


def run_probabilistic_regression_models(
    X, y, prop_name: str, num_folds: int = 5, save_path: str = None,
    fig_dir_path: str = None
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

        """
    print(f"\n[INFO] Running CV for property: {prop_name} with {num_folds} folds")
    print(f"[INFO] Total models: {len(probabilistic_regression_models)}")

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Generating Input Shape for Tensorflow model
    input_shape = X.shape[1:]

    # Storage for results
    results = {name: {"R2_train": [], "R2_test": [],
                      "MAE_train": [], "MAE_test": [],
                      "MSE_train": [], "MSE_test": [],
                      "PICP_train": [], "PICP_test": [],
                      "MPIW_train": [], "MPIW_test": [],
                      "STD_train_mean": [], "STD_test_mean": [],
                      }
               for name in probabilistic_regression_names}

    # Iterate through each fold
    for fold_num, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print(f"\n[INFO] Fold {fold_num}/{num_folds}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

            # Collect metrics
            results[model_name]["R2_train"].append(r2_score(y_train, mean_tr))
            results[model_name]["R2_test"].append(r2_score(y_test, mean_te))
            results[model_name]["MAE_train"].append(mean_absolute_error(y_train, mean_tr))
            results[model_name]["MAE_test"].append(mean_absolute_error(y_test, mean_te))
            results[model_name]["MSE_train"].append(mean_squared_error(y_train, mean_tr))
            results[model_name]["MSE_test"].append(mean_squared_error(y_test, mean_te))

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
    summary_rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric, values in metrics.items():
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Model")

    # Save CSV if requested
    if save_path:
        filename = f"{save_path}/{prop_name}_cv_results.csv"
        summary_df.to_csv(filename)
        print(f"[INFO] Results saved to {filename}")

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