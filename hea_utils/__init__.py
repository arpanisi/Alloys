from .regression_models import (tf_regression_model, tf_bnn_regression_model,
                                tf_bnn_regression_vi, tf_prob_regression_model)
from .data_preparation import (load_data, load_oxidation_data,
                               load_training_data, prepare_training_data)
from .run_regression_models import (run_regular_regression_models, run_ensemble_regression_models,
                                    run_probabilistic_regression_models)

from .run_regression_models import (classical_regression_models, ensemble_regression_models,
                                    probabilistic_regression_models)
from .run_regression_models import (classical_regression_names, ensemble_regression_names,
                                    probabilistic_regression_names)

tf_models = [tf_regression_model,
             tf_bnn_regression_model, tf_bnn_regression_vi,
             tf_prob_regression_model]

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)',
         'Oxidation (mass change_per_hr)']

# Combine names and models
all_names = classical_regression_names + ensemble_regression_names + probabilistic_regression_names
all_models = classical_regression_models + ensemble_regression_models + probabilistic_regression_models

# Create dict mapping from name → model
all_models_dict = dict(zip(all_names, all_models))
