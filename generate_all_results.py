from hea_utils import (run_regular_regression_models,
                       run_ensemble_regression_models,
                       run_probabilistic_regression_models)
from hea_utils import load_training_data, prepare_training_data
from hea_utils import props

from sklearn.model_selection import train_test_split

# Loop through each property in the list `props`
for prop_ind, prop in enumerate(props):

    X, y, Z = load_training_data(prop_ind, prop)

    X, y, lbe, std, synthetic_alloys = prepare_training_data(X, y, Z, prop_ind, prop)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Running regular regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/regular_regressions'
    run_regular_regression_models(X, y, prop_name=prop, save_path=save_dir_path)

    # Running regular regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/ensemble_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/ensemble_regressions'
    run_ensemble_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                   fig_dir_path=fig_dir_path)

    # Running probabilistic regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/probabilistic_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/probabilistic_regressions'
    run_probabilistic_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                   fig_dir_path=fig_dir_path)


for prop_ind, prop in enumerate(props):

    X, y, _ = load_training_data(prop_ind, prop)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Running regular regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/comp_only/regular_regressions'
    run_regular_regression_models(X, y, prop_name=prop, save_path=save_dir_path)

    # Running regular regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/comp_only/ensemble_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/comp_only/ensemble_regressions'
    run_ensemble_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                   fig_dir_path=fig_dir_path)

    # Running probabilistic regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/comp_only/probabilistic_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/comp_only/probabilistic_regressions'
    run_probabilistic_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                   fig_dir_path=fig_dir_path)

# One Loop optimization
from hea_utils.regression_models import tf_bnn_regression_model
import pandas as pd
import numpy as np
from hea_utils.pareto_optimization import is_dominated
synth_props = {}

for prop_ind, prop in enumerate(props):

    X, y, Z = load_training_data(prop_ind, prop)

    X, y, lbe, std, synthetic_alloys_pred = prepare_training_data(X, y, Z, prop_ind, prop)
    synthetic_alloys_pred = synthetic_alloys_pred[X.columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    input_size = len(X_train)
    input_shape = X.shape[1:]
    model = tf_bnn_regression_model(input_size, input_shape)

    model.fit(X_train, y_train, epochs=4000)

    y_pred = model(synthetic_alloys_pred.values)
    mean_predictions = y_pred.mean().numpy().flatten()

    synth_props[prop] = mean_predictions

synth_props_df = pd.DataFrame(synth_props)
synth_props_df[props[-1]] = -synth_props_df[props[-1]]

pareto_front = []
inds = []
for i, row in synth_props_df.iterrows():
    if not is_dominated(row, synth_props_df):
        pareto_front.append(row)
        inds.append(i)

pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df[props[-1]] = -pareto_front_df[props[-1]]

synthetic_alloys = pd.read_csv('data/synthetic_alloys.csv', index_col=0)
pareto_alloys = synthetic_alloys.iloc[inds]