import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_context('talk')
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from exercises.data_preparation import load_complete_data, load_data
from sklearn.model_selection import train_test_split

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
X, y, Z = load_data(col=props[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define the probabilistic regression model
def create_probabilistic_regression_model(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(20, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(2),  # Two outputs for mean and standard deviation
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(t[..., 1:]))),
    ])
    return model

# Create the probabilistic regression model
model = create_probabilistic_regression_model(input_shape=X.shape[1:])

# Compile the model
def negative_log_likelihood(y_true, y_pred):
    return -tf.reduce_mean(y_pred.log_prob(y_true))


model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negative_log_likelihood)

# Train the model
model.fit(X_train, y_train, epochs=1000)

# Evaluate the model
y_pred = model(X_test.values)

# Extract mean and standard deviation from the output distribution
mean_predictions = y_pred.mean().numpy()
stddev_predictions = y_pred.stddev().numpy()

lower_limit = mean_predictions - 1.96 * stddev_predictions
upper_limit = mean_predictions + 1.96 * stddev_predictions

CI = pd.DataFrame({'Observed': y_test.values, 'CI_lower': lower_limit.flatten(),
                   'CI_upper': upper_limit.flatten()})
plt.scatter(np.arange(len(y_test)), y_test)
plt.fill_between(np.arange(len(y_test)),
                 lower_limit.flatten(), upper_limit.flatten(),
                 color='r', alpha=0.3, label="95% Confidence Interval")
plt.plot(mean_predictions)
plt.show()