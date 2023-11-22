import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('talk')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from exercises.data_preparation import load_oxidation_data
from sklearn.model_selection import train_test_split
from exercises.regression_models import tf_prob_regression_model, tf_bnn_regression_model
from sklearn.metrics import r2_score

elem_comp, synth_data, y = load_oxidation_data()

# Loading Synthetic Alloys
synthetic_alloys = pd.read_csv('../data/synthetic_alloys.csv')
synthetic_comps = synthetic_alloys[elem_comp.columns]
synthetic_conditions = synthetic_alloys[synth_data.columns]

lbe = LabelEncoder()
std = StandardScaler()

for col in synth_data.columns[:2]:
    synthetic_conditions[col] = lbe.fit_transform(synthetic_conditions[col])
    synth_data[col] = lbe.transform(synth_data[col])

Z_scaled = pd.DataFrame(std.fit_transform(synth_data),
                        columns=synth_data.columns, index=synth_data.index)
synth_scaled = pd.DataFrame(std.transform(synthetic_conditions),
                        columns=synthetic_conditions.columns, index=synthetic_conditions.index)

X = pd.concat([elem_comp, Z_scaled], axis=1)
synthetic_alloys = pd.concat([synthetic_comps, synth_scaled], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)

input_size = len(X_train)
input_shape = X_train.shape[1:]

model = tf_bnn_regression_model(input_data_size=input_size,
                                input_shape=input_shape)

model.fit(X_train, y_train, epochs=4000)

# Evaluate the model
y_pred = model(X_test.values)

# Extract mean and standard deviation from the output distribution
mean_predictions = y_pred.mean().numpy()
stddev_predictions = y_pred.stddev().numpy()

lower_limit = mean_predictions - 1.96 * stddev_predictions
upper_limit = mean_predictions + 1.96 * stddev_predictions

CI_bnn = pd.DataFrame({'Observed': y_test.values, 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
plt.scatter(np.arange(len(y_test)), y_test)
plt.fill_between(np.arange(len(y_test)),
                 lower_limit.flatten(), upper_limit.flatten(),
                 color='r', alpha=0.3, label="95% Confidence Interval")
plt.plot(mean_predictions)
plt.title('Oxidation State')
plt.savefig(f'../figs/bnn_regression_oxidation.png', bbox_inches='tight', dpi=300)
plt.show()

y_pred = model(synthetic_alloys.values)
# Extract mean and standard deviation from the output distribution
mean_predictions = y_pred.mean().numpy()
stddev_predictions = y_pred.stddev().numpy()

lower_limit = mean_predictions - 1.96 * stddev_predictions
upper_limit = mean_predictions + 1.96 * stddev_predictions

CI_bnn_synth = pd.DataFrame({'Mean': mean_predictions.flatten(), 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
CI_bnn_synth.to_csv(f'../data/synthetic_prediction_oxidation_bnn.csv')

model = tf_prob_regression_model(input_shape=X.shape[1:])
model.fit(X_train, y_train, epochs=1000)

# Evaluate the model
y_pred = model(X_test.values)

# Extract mean and standard deviation from the output distribution
mean_predictions = y_pred.mean().numpy()
stddev_predictions = y_pred.stddev().numpy()

lower_limit = mean_predictions - 1.96 * stddev_predictions
upper_limit = mean_predictions + 1.96 * stddev_predictions

CI_pnn = pd.DataFrame({'Observed': y_test.values, 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
plt.scatter(np.arange(len(y_test)), y_test)
plt.fill_between(np.arange(len(y_test)),
                 lower_limit.flatten(), upper_limit.flatten(),
                 color='r', alpha=0.3, label="95% Confidence Interval")
plt.plot(mean_predictions)
plt.title('Oxidation State')
plt.savefig(f'../figs/pnn_regression_oxidation.png', bbox_inches='tight', dpi=300)
plt.show()

y_pred = model(synthetic_alloys.values)
# Extract mean and standard deviation from the output distribution
mean_predictions = y_pred.mean().numpy()
stddev_predictions = y_pred.stddev().numpy()

lower_limit = mean_predictions - 1.96 * stddev_predictions
upper_limit = mean_predictions + 1.96 * stddev_predictions

CI_bnn_synth = pd.DataFrame({'Mean': mean_predictions.flatten(), 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
CI_bnn_synth.to_csv(f'../data/synthetic_prediction_oxidation_pnn.csv')