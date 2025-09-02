import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import PassiveAggressiveRegressor, RANSACRegressor, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor, PoissonRegressor, GammaRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.ensemble import ExtraTreesRegressor
import tf_keras as keras

# Initializing regression models
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=0.1)
elastic_net_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
svr_reg = SVR(kernel='rbf')
nu_svr_reg = NuSVR(kernel='rbf')
linear_svr_reg = LinearSVR(epsilon=0.0)
decision_tree_reg = DecisionTreeRegressor()
random_forest_reg = RandomForestRegressor(random_state=42)
gradient_boosting_reg = GradientBoostingRegressor(random_state=42)
ada_boost_reg = AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42)
bagging_reg = BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=42)
knn_reg = KNeighborsRegressor(n_neighbors=5)
radius_neighbors_reg = RadiusNeighborsRegressor(radius=1.0)
kernel_ridge_reg = KernelRidge(alpha=1.0, kernel='linear')
mlp_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
isotonic_reg = IsotonicRegression()
passive_aggressive_reg = PassiveAggressiveRegressor(C=1.0, epsilon=0.1, random_state=42)
ransac_reg = RANSACRegressor()
huber_reg = HuberRegressor(epsilon=1.35)
hist_gradient_boosting_reg = HistGradientBoostingRegressor(random_state=42)
tweedie_reg = TweedieRegressor(power=0, alpha=0.5, link='auto')
poisson_reg = PoissonRegressor()
gamma_reg = GammaRegressor()
gaussian_process_reg = GaussianProcessRegressor(random_state=42)
voting_reg = VotingRegressor(estimators=[('lr', linear_reg), ('rf', random_forest_reg)])
theil_sen_reg = TheilSenRegressor(random_state=42)
extra_trees_reg = ExtraTreesRegressor(random_state=42)
import tensorflow as tf
import tensorflow_probability as tfp


def tf_regression_model(input_shape):

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
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='Adam', loss='mse')
    return model


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
          tfp.distributions.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
          tfp.distributions.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])


def tf_bnn_regression_model(input_data_size, input_shape):

    model = keras.Sequential([      # <-- keras.Sequential from tf_keras
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        tfp.layers.DenseVariational(  # Probabilistic dense layer
            units=1 + 1,  # Output dimension
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior_trainable,
            kl_weight=1 / input_data_size,
        ),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.StudentT(df=5, loc=t[..., :1],
                        scale=1e-3 + tf.math.softplus(t[..., 1:]))),
    ])

    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer='Adam', loss=negloglik)
    return model


def tf_bnn_regression_vi(input_data_size, input_shape):

    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        tfp.layers.DenseFlipout(20, activation='relu'),
        keras.layers.Activation('relu'),
        tfp.layers.DenseFlipout(20, activation='relu'),
        keras.layers.Activation('relu'),
        tfp.layers.DenseFlipout(20, activation='relu'),
        keras.layers.Activation('relu'),
        tfp.layers.DenseVariational(  # Probabilistic dense layer
            units=1 + 1,  # Output dimension
            make_posterior_fn=posterior_mean_field,
            make_prior_fn=prior_trainable,
            kl_weight=1 / input_data_size,
        ),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:]))),
    ])

    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer='Adam', loss=negloglik)
    return model


def tf_prob_regression_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(20, activation='relu'),
        keras.layers.Dense(2),  # Two outputs for mean and standard deviation
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.StudentT(df=5, loc=t[..., :1],
                            scale=1e-3 + tf.math.softplus(t[..., 1:]))),
    ])

    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer='Adam', loss=negloglik)
    return model
