from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np


def regression_metrics(regression_names, prob_model=False):

    # Define metrics
    base_metrics = ["R2_train", "R2_test",
                    "MAE_train", "MAE_test",
                    "MSE_train", "MSE_test"]

    extra_metrics = ["PICP_train", "PICP_test",
                     "MPIW_train", "MPIW_test",
                     "STD_train_mean", "STD_test_mean"]
    # Storage for results
    results = {}

    for name in regression_names:
        metrics = base_metrics.copy()
        if prob_model:
            metrics += extra_metrics
        results[name] = {m: [] for m in metrics}

    return results


def compute_regression_metrics(y_train, y_pred_train, y_test, y_pred_test):
    """
    Compute regression metrics for train and test sets.

    Args:
        y_train (np.ndarray or pd.Series): True training values.
        y_pred_train (np.ndarray): Predicted training values.
        y_test (np.ndarray or pd.Series): True test values.
        y_pred_test (np.ndarray): Predicted test values.

    Returns:
        dict: Dictionary with R2, MAE, and MSE for train and test.
    """
    return {
        "R2_train": r2_score(y_train, y_pred_train),
        "R2_test": r2_score(y_test, y_pred_test),
        "MAE_train": mean_absolute_error(y_train, y_pred_train),
        "MAE_test": mean_absolute_error(y_test, y_pred_test),
        "MSE_train": mean_squared_error(y_train, y_pred_train),
        "MSE_test": mean_squared_error(y_test, y_pred_test),
    }


def summarize_cv_results(results, prop_name, save_path=None):
    """
    Summarize cross-validation results into a DataFrame with mean ± std.

    Args:
        results (dict): Nested dictionary with structure
                        {model_name: {metric: [values_across_folds]}}.
        prop_name (str): Name of the property (used for saving).
        save_path (str or None): Directory to save CSV summary, or None to skip saving.

    Returns:
        pd.DataFrame: Summary table with mean and std for each metric.
    """
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
