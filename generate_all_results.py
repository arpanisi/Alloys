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
    save_dir_path = 'models/models_11_1_2025/comp_plus_process/regular_regressions'
    model_save_path = 'models/models_11_1_2025/comp_plus_process/regular_regressions'
    run_regular_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                  model_save_path=model_save_path)

    # Running regular regression models and generating tabular report
    save_dir_path = 'models/models_11_1_2025/comp_plus_process/ensemble_regressions'
    model_save_path = 'models/models_11_1_2025/comp_plus_process/ensemble_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/comp_plus_process/ensemble_regressions'
    run_ensemble_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                   fig_dir_path=fig_dir_path,
                                   model_save_path=model_save_path)

    # Running probabilistic regression models and generating tabular report
    save_dir_path = 'models/models_11_1_2025/comp_plus_process/probabilistic_regressions'
    model_save_path = 'models/models_11_1_2025/comp_plus_process/probabilistic_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/comp_plus_process/probabilistic_regressions'
    run_probabilistic_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                        fig_dir_path=fig_dir_path,
                                        model_save_path=model_save_path)

for prop_ind, prop in enumerate(props):
    X, y, _ = load_training_data(prop_ind, prop)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # Running regular regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/comp_only/regular_regressions'
    model_save_path = 'models/models_11_1_2025/comp_only/regular_regressions'
    run_regular_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                  model_save_path=model_save_path)

    # Running regular regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/comp_only/ensemble_regressions'
    model_save_path = 'models/models_11_1_2025/comp_only/ensemble_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/comp_only/ensemble_regressions'
    run_ensemble_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                   fig_dir_path=fig_dir_path, model_save_path=model_save_path)

    # Running probabilistic regression models and generating tabular report
    save_dir_path = 'results/results_11_1_2025/comp_only/probabilistic_regressions'
    fig_dir_path = 'figs/figs_11_1_2025/uncertainty_plots/comp_only/probabilistic_regressions'
    model_save_path = 'models/models_11_1_2025/comp_only/probabilistic_regressions'
    run_probabilistic_regression_models(X, y, prop_name=prop, save_path=save_dir_path,
                                        fig_dir_path=fig_dir_path,
                                        model_save_path=model_save_path)


