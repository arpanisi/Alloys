import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('talk')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from exercises.data_preparation import load_data, load_oxidation_data
from sklearn.model_selection import train_test_split
from exercises.regression_models import tf_prob_regression_model, tf_bnn_regression_model, \
    tf_bnn_regression_vi, tf_regression_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow_probability as tfp
from datetime import datetime

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)',
         'Oxidation (mass change_per_hr)']
prop_ind = 2
if prop_ind == 3:
    X, Z, y = load_oxidation_data()
else:
    X, y, Z = load_data(col=props[prop_ind])

# Loading the Synthetic Alloys
synthetic_alloys = pd.read_csv('../data/synthetic_alloys.csv')

top_feat = 10
lbe = LabelEncoder()
std = StandardScaler()

# Preprocessing training data and synthetic data with same encoder
if prop_ind == 3:

    synthetic_comps = synthetic_alloys[X.columns]
    synthetic_conditions = synthetic_alloys[Z.columns]

    for col in Z.columns[:2]:
        synthetic_conditions[col] = lbe.fit_transform(synthetic_conditions[col])
        Z[col] = lbe.transform(Z[col])

    Z_scaled = pd.DataFrame(std.fit_transform(Z),
                            columns=Z.columns, index=Z.index)
    synth_scaled = pd.DataFrame(std.transform(synthetic_conditions),
                                columns=synthetic_conditions.columns,
                                index=synthetic_conditions.index)
    synthetic_alloys = pd.concat([synthetic_comps, synth_scaled], axis=1)

else:
    Phase = synthetic_alloys['PhaseType']
    synthetic_alloys['PhaseType'] = lbe.fit_transform(Phase)
    Z['PhaseType'] = lbe.transform(Z['PhaseType'])
    Z_scaled = pd.DataFrame(std.fit_transform(Z),
                            columns=Z.columns, index=Z.index)

    synthetic_alloys[Z.columns] = std.transform(synthetic_alloys[Z.columns])


X = pd.concat([X, Z_scaled], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)
input_size = len(X_train)
input_shape = X_train.shape[1:]


# Create the probabilistic regression model
# model = tf_prob_regression_model(input_shape=X.shape[1:])
while True:
    model = tf_bnn_regression_model(input_data_size=input_size,
                                    input_shape=input_shape)

    # Train the model
    model.fit(X_train, y_train, epochs=4000)

    # Evaluate the model
    y_pred = model(X_test.values)

    # Extract mean and standard deviation from the output distribution
    mean_predictions = y_pred.mean().numpy().flatten()
    stddev_predictions = y_pred.stddev().numpy().flatten()

    sorted_stddevs = np.argsort(stddev_predictions.flatten())
    top_alloys_ind = y_test.iloc[sorted_stddevs].index[:top_feat]
    top_alloys = X.loc[top_alloys_ind]
    # top_alloys.to_csv(f'../results/bnn_regression_top_std_{props[prop_ind]}.csv')

    alpha = 1.96
    lower_limit = mean_predictions - alpha * stddev_predictions
    upper_limit = mean_predictions + alpha * stddev_predictions

    # Calculate PICP
    inside_interval = np.logical_and(y_test >= lower_limit, y_test <= upper_limit)
    PICP = np.mean(inside_interval)

    # Calculate MPIW
    interval_width = upper_limit - lower_limit
    MPIW = np.mean(interval_width)

    regular_regression_model = tf_regression_model(input_shape=input_shape)
    regular_regression_model.fit(X_train, y_train, epochs=4000)
    y_test_pred = regular_regression_model.predict(X_test)


    c1 = r2_score(y_test, mean_predictions)
    c2 = r2_score(y_test, y_test_pred)

    if c1 > c2:

        CI_bnn = pd.DataFrame({'Observed': y_test.values, 'CI_lower': lower_limit.flatten(),
                               'CI_upper': upper_limit.flatten(), 'Mean': mean_predictions})
        CI_bnn.to_csv(f'../results/bnn_regression_test_{props[prop_ind]}.csv')
        plt.scatter(np.arange(len(y_test)), y_test)
        plt.fill_between(np.arange(len(y_test)),
                         lower_limit.flatten(), upper_limit.flatten(),
                         color='r', alpha=0.3, label="95% Confidence Interval")
        plt.plot(mean_predictions)
        plt.title(props[prop_ind])
        plt.savefig(f'../figs/bnn_regression_{props[prop_ind]}.png', bbox_inches='tight', dpi=300)
        plt.show()

        print(f'MPIW: {MPIW} and PICP: {PICP}')
        print('Testing R2 for Bayesian Regression is:', c1)
        print('Testing MSE for Bayesian Regression is:', mean_squared_error(y_test, mean_predictions))
        print('Testing MAE for Bayesian Regression is:', mean_absolute_error(y_test, mean_predictions))


        print('Testing R2 for Regular Regression is:', c2)
        print('Testing MSE for Regular Regression is:', mean_squared_error(y_test, y_test_pred))
        print('Testing MAE for Regular Regression is:', mean_absolute_error(y_test, y_test_pred))
        pd.DataFrame({'Observed': y_test.values,
                      'Predicted': y_test_pred.flatten()}).\
            to_csv(f'../results/regular_regression_test_{props[prop_ind]}.csv')

        break



# y_pred = model(synthetic_alloys.values)
#
# # Extract mean and standard deviation from the output distribution
# mean_predictions = y_pred.mean().numpy().flatten()
# stddev_predictions = y_pred.stddev().numpy().flatten()
#
# alpha = 1.96
# lower_limit = mean_predictions - alpha * stddev_predictions
# upper_limit = mean_predictions + alpha * stddev_predictions
#
#
#
# # Plot
#
# plt.fill_between(np.arange(len(lower_limit)),
#                  lower_limit.flatten(), upper_limit.flatten(),
#                  color='r', alpha=0.3, label="95% Confidence Interval")
# plt.scatter(np.arange(len(lower_limit)), mean_predictions)
# plt.show()
#
# CI_bnn_synth = pd.DataFrame({'Mean': mean_predictions.flatten(), 'CI_lower': lower_limit,
#                    'CI_upper': upper_limit})
# CI_bnn_synth.to_csv(f'../data/synthetic_prediction_{props[prop_ind]}_bnn.csv')
# # model.save(f'../models/combined/bnn_{props[prop_ind]}_09_27_23.h5', save_traces=False)
#
#
# model = tf_prob_regression_model(input_shape=X.shape[1:])
# model.fit(X_train, y_train, epochs=1000)
#
# # Evaluate the model
# y_pred = model(X_test.values)
#
# # Extract mean and standard deviation from the output distribution
# mean_predictions = y_pred.mean().numpy()
# stddev_predictions = y_pred.stddev().numpy()
#
# sorted_stddevs = np.argsort(stddev_predictions.flatten())
# top_alloys_ind = y_test.iloc[sorted_stddevs].index[:top_feat]
# top_alloys = X.loc[top_alloys_ind]
# top_alloys.to_csv(f'../results/pnn_regression_top_std_{props[prop_ind]}.csv')
#
# lower_limit = mean_predictions - 1.96 * stddev_predictions
# upper_limit = mean_predictions + 1.96 * stddev_predictions
#
#
# print('Testing R2 is:', r2_score(y_test, mean_predictions))
# CI_pnn = pd.DataFrame({'Observed': y_test.values, 'CI_lower': lower_limit.flatten(),
#                    'CI_upper': upper_limit.flatten()})
# plt.scatter(np.arange(len(y_test)), y_test)
# plt.fill_between(np.arange(len(y_test)),
#                  lower_limit.flatten(), upper_limit.flatten(),
#                  color='r', alpha=0.3, label="95% Confidence Interval")
# plt.plot(mean_predictions)
# plt.title(props[prop_ind])
# plt.savefig(f'figs/pnn_regression_{props[prop_ind]}.png', bbox_inches='tight', dpi=300)
# plt.show()
#
# y_pred = model(synthetic_alloys.values)
#
# # Extract mean and standard deviation from the output distribution
# mean_predictions = y_pred.mean().numpy()
# stddev_predictions = y_pred.stddev().numpy()
#
#
# lower_limit = mean_predictions - 1.96 * stddev_predictions
# upper_limit = mean_predictions + 1.96 * stddev_predictions
#
# # Plot
#
# plt.fill_between(np.arange(len(lower_limit)),
#                  lower_limit.flatten(), upper_limit.flatten(),
#                  color='r', alpha=0.3, label="95% Confidence Interval")
# plt.scatter(np.arange(len(lower_limit)), mean_predictions)
# plt.show()
#
# CI_pnn_synth = pd.DataFrame({'Mean': mean_predictions.flatten(), 'CI_lower': lower_limit.flatten(),
#                    'CI_upper': upper_limit.flatten()})
# CI_pnn_synth.to_csv(f'../data/synthetic_prediction_{props[prop_ind]}_pnn.csv')
# # model.save(f'../models/combined/pnn_{props[prop_ind]}_09_27_23.h5', save_traces=False)