from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns

import tensorflow as tf

import tensorflow_probability as tfp

sns.reset_defaults()
sns.set_context(context='talk',font_scale=0.7)

tfd = tfp.distributions
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

X = np.random.random((100, 10))
y = np.random.random(100)
x_test = np.random.random((1000, 10))
input_shape = X.shape[1:]

def my_dist(params):
 return tfp.distributions.Normal(loc=params, scale=1)

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
    tf.keras.layers.Dropout(0.3)]
)

params = tf.keras.layers.Dense(1)(model.output)

dist = tfp.layers.DistributionLambda(my_dist)(params)
model = tf.keras.models.Model(inputs=model.inputs, outputs=dist)
# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss='mse')
history = model.fit(X, y, epochs=1000)

# Profit.
yhat = model(x_test)



