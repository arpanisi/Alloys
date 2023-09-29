import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('talk')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from exercises.data_preparation import load_complete_data, load_data, synthetic_data
from sklearn.model_selection import train_test_split
from exercises.regression_models import tf_prob_regression_model, tf_bnn_regression_model

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
prop_ind = 2
X, y, Z = load_data(col=props[prop_ind])
Phase = Z['PhaseType']

lbe = LabelEncoder()
std = StandardScaler()
Phase_labeled = lbe.fit_transform(Phase)
Z['PhaseType'] = Phase_labeled

Z_scaled = pd.DataFrame(std.fit_transform(Z),
                        columns=Z.columns, index=Z.index)
X = pd.concat([X, Z_scaled], axis=1)

#synthetic data generating
synthetic_alloys = pd.read_csv('../data/synthetic_alloys.csv')
synthetic_alloys = synthetic_alloys[X.columns]
synthetic_alloys['PhaseType'] = lbe.transform(synthetic_alloys['PhaseType'])
synthetic_alloys[Z.columns] = std.transform(synthetic_alloys[Z.columns])

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


y_pred = model(synthetic_alloys.values)

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

CI_bnn_synth = pd.DataFrame({'Mean': mean_predictions.flatten(), 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
CI_bnn_synth.to_csv(f'../data/synthetic_prediction_{props[prop_ind]}_bnn.csv')
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

y_pred = model(synthetic_alloys.values)

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

CI_pnn_synth = pd.DataFrame({'Mean': mean_predictions.flatten(), 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
CI_pnn_synth.to_csv(f'../data/synthetic_prediction_{props[prop_ind]}_pnn.csv')
# model.save(f'models/combined/pnn_{props[prop_ind]}_09_27_23.h5')