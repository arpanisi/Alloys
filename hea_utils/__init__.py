from .regression_models import (tf_regression_model, tf_bnn_regression_model,
                                tf_bnn_regression_vi, tf_prob_regression_model)
from .data_preparation import (load_data, load_oxidation_data,
                               load_training_data, prepare_training_data)

from .run_regression_models import run_regular_regression_models, run_ensemble_regression_models

tf_models = [tf_regression_model,
           tf_bnn_regression_model, tf_bnn_regression_vi,
           tf_prob_regression_model]

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)',
         'Oxidation (mass change_per_hr)']

