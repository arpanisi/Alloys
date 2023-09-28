import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('talk')
from sklearn.preprocessing import StandardScaler
from exercises.data_preparation import load_complete_data, load_data, synthetic_data
from sklearn.model_selection import train_test_split
from exercises.regression_models import tf_prob_regression_model, tf_bnn_regression_model

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
prop_ind = 2
X, y, Z = load_data(col=props[prop_ind])

std = StandardScaler()
Z_scaled = pd.DataFrame(std.fit_transform(Z),
                        columns=Z.columns, index=Z.index)
X = pd.concat([X, Z_scaled], axis=1)

#synthetic data generating
X_synth, Z_synth = synthetic_data(col=props[prop_ind])
Z_synth_scaled = std.transform(Z_synth)
X_tot_synth = pd.concat([X_synth, Z_synth], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)
input_size = len(X_train)
input_shape = X_train.shape[1:]


# Create the probabilistic regression model
# model = tf_prob_regression_model(input_shape=X.shape[1:])
model = tf_bnn_regression_model(input_data_size=input_size,
                                input_shape=input_shape)

# Train the model
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
plt.show()


y_pred = model(X_tot_synth.values)

# Extract mean and standard deviation from the output distribution
mean_predictions = y_pred.mean().numpy()
stddev_predictions = y_pred.stddev().numpy()

lower_limit = mean_predictions - 1.96 * stddev_predictions
upper_limit = mean_predictions + 1.96 * stddev_predictions

# Plot

plt.fill_between(np.arange(len(lower_limit)),
                 lower_limit.flatten(), upper_limit.flatten(),
                 color='r', alpha=0.3, label="95% Confidence Interval")
plt.scatter(np.arange(len(lower_limit)), mean_predictions)
plt.show()

# model.save(f'models/combined/bnn_{props[prop_ind]}_09_27_23.h5')


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
plt.show()

# model.save(f'models/combined/pnn_{props[prop_ind]}_09_27_23.h5')