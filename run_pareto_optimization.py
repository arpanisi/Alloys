from hea_utils import all_models_dict
import pandas as pd
from hea_utils.pareto_optimization import is_dominated
from hea_utils import load_training_data, prepare_training_data
from hea_utils import props
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')

synth_props = {}
model_name = "Linear Regression"
fold_num = None
ablation_study = 'comp_only'
model_state = 'pretrained'
tf_model = 0
tfp_model = 0

model_fn = all_models_dict[model_name]


for prop_ind, prop in enumerate(props):

    X, y, Z = load_training_data(prop_ind, prop)
    elem_comp = X.columns

    if ablation_study == 'comp_only':
        synthetic_alloys = pd.read_csv('data/synthetic_alloys.csv', index_col=0)

    if ablation_study == 'comp_plus_process':
        X, y, lbe, std, synthetic_alloys = prepare_training_data(X, y, Z, prop_ind, prop)

    synthetic_alloys_pred = synthetic_alloys[X.columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    input_size = len(X_train)
    input_shape = X.shape[1:]

    if model_name == "Deep Neural Network Regression":
        model = model_fn(input_shape)
        tf_model = 1
    if model_name == "BNN-Variational Regression" or model_name == "BNN-Flipout Regression":
        model = model_fn(input_data_size=input_size, input_shape=input_shape)
        tf_model = 1
        tfp_model = 1
    else:
        model = model_fn

    if tf_model:
        model.fit(X_train, y_train, epochs=4000)
    else:
        model.fit(X_train, y_train)

    if tfp_model:
        y_pred = model(synthetic_alloys_pred.values)
        mean_predictions = y_pred.mean().numpy().flatten()
    else:
        mean_predictions = model.predict(synthetic_alloys_pred.values)

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

pareto_compositions = pareto_alloys[elem_comp]
plt.figure(figsize=(10, 8))
sns.boxplot(pareto_compositions * 100)

plt.xlabel("Element")
plt.ylabel("Atomic Fraction (%)")
plt.title(f"Distribution of Elemental Compositions in Pareto Alloys using\n{model_name}")
plt.tight_layout()
plt.show()