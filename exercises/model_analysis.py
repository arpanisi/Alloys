import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from exercises.regression_models import *
from exercises.data_preparation import load_data
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
prop_ind = 0
X, y, _ = load_data(col=props[prop_ind])

y_max = y.max()
y = y/y_max


# X = pd.concat([X, Z_scaled], axis=1)
# X = Z_scaled.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

input_shape = X.shape[1:]
input_data_size = X.shape[0]


model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(1 + 1),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),
])


model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu', activity_regularizer='l2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu', activity_regularizer='l2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu', activity_regularizer='l2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu', activity_regularizer='l2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', activity_regularizer='l2'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', activity_regularizer='l2'),
        tf.keras.layers.Dropout(0.3),
        # tf.keras.layers.Dense(1),
        # tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t, scale=1))
        tfp.layers.DenseVariational(  # Probabilistic dense layer
            units=1,  # Output dimension
            make_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            make_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kl_weight=1 / input_data_size,
            activation=None,  # Choose activation function
        ),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t, scale=1))
    ])

negloglik = lambda y, rv_y: -rv_y.log_prob(y)
model.compile(optimizer='Adam', loss=negloglik)

for i in range(10):

    kernel = 1.0 * RBF(length_scale=1.0)  # You can choose an appropriate kernel
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(X_train, y_train, epochs=100)

    synthetic_alloys = pd.DataFrame(np.random.random((100, X.shape[1])), columns=X.columns)
    synthetic_alloys = synthetic_alloys.div(synthetic_alloys.sum(axis=1), axis=0)

    r2_score(y_test, model.predict(X_test))
    y_pred, y_std = model.predict(synthetic_alloys, return_std=True)
    entropy = -0.5 * np.log(2 * np.pi * np.e * y_std)


plt.plot(history.epoch, history.history['loss'], c='k', label='Training')
plt.plot(history.epoch, history.history['val_loss'], c='r', label='Testing')
plt.legend()
plt.show()

train_r2 = r2_score(y_train, model.predict(X_train))
test_r2 = r2_score(y_test, model.predict(X_test))
plt.scatter(y_train, model.predict(X_train), c='k',
            label=f"Training R2: {train_r2:.4f}")
plt.scatter(y_test, model.predict(X_test), c='r',
            label=f"Testing R2: {test_r2:.4f}")

plt.legend()
plt.show()

print("R-squared score (Test):", test_r2)
print("R-squared score (Training):", train_r2)

from datetime import datetime
model_name = 'Regression_' + props[prop_ind] + '_' + datetime.today().strftime('%m_%d_%Y') + '.h5'
model.save('../models/' + model_name)