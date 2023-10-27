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

lbe = LabelEncoder()
std = StandardScaler()

for col in synth_data.columns[:2]:
    synth_data[col] = lbe.fit_transform(synth_data[col])

Z_scaled = pd.DataFrame(std.fit_transform(synth_data),
                        columns=synth_data.columns, index=synth_data.index)

X = pd.concat([elem_comp, Z_scaled], axis=1)

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
print('')

lower_limit = mean_predictions - 1.96 * stddev_predictions
upper_limit = mean_predictions + 1.96 * stddev_predictions

CI_bnn = pd.DataFrame({'Observed': y_test.values, 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
plt.scatter(np.arange(len(y_test)), y_test)
plt.fill_between(np.arange(len(y_test)),
                 lower_limit.flatten(), upper_limit.flatten(),
                 color='r', alpha=0.3, label="95% Confidence Interval")
plt.plot(mean_predictions)
plt.title('Bayesian Regression')
plt.savefig(f'../figs/bnn_regression_oxidation.png', bbox_inches='tight', dpi=300)
plt.show()

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
plt.title('Probabilistic Regression')
plt.savefig(f'../figs/pnn_regression_oxidation.png', bbox_inches='tight', dpi=300)
plt.show()