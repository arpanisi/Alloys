from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
from exercises.data_preparation import *


import tensorflow as tf
import tensorflow_probability as tfp

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
prop_ind = 0
X, y, _ = load_data(col=props[prop_ind])
y_max = y.max()
y = y / y_max
sns.reset_defaults()
sns.set_context(context='talk', font_scale=0.7)

tfd = tfp.distributions
negloglik = lambda y, rv_y: -rv_y.log_prob(y)

synthetic_alloys = pd.DataFrame(np.random.random((100, X.shape[1])), columns=X.columns)
synthetic_alloys = synthetic_alloys.div(synthetic_alloys.sum(axis=1), axis=0)

x_test = synthetic_alloys.copy()
input_shape = X.shape[1:]

def my_dist(params):
 return tfp.distributions.Normal(loc=params, scale=1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tfp.layers.DenseReparameterization(1 + 1),  # Output layer (mean + variance)
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))
    )  # Create a Normal distribution with mean and variance
])

# params = tf.keras.layers.Dense(1)(model.output)
#
# dist = tfp.layers.DistributionLambda(my_dist)(params)
# model = tf.keras.models.Model(inputs=model.inputs, outputs=dist)
# Do inference.
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negloglik)
history = model.fit(X, y, epochs=200)

# Profit.
yhat = model(X.values)
y_pred = model.predict(synthetic_alloys)
y_mean = yhat.mean().numpy()
y_var = yhat.variance().numpy()



