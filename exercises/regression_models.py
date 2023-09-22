import numpy as np, pandas as pd
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
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF

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
ada_boost_reg = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), random_state=42)
bagging_reg = BaggingRegressor(base_estimator=DecisionTreeRegressor(), random_state=42)
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


# List of models
regression_models = [linear_reg, ridge_reg, lasso_reg, elastic_net_reg,
                     svr_reg, nu_svr_reg, linear_svr_reg, decision_tree_reg,
                     random_forest_reg, gradient_boosting_reg, ada_boost_reg,
                     bagging_reg, knn_reg, kernel_ridge_reg,
                     mlp_reg, passive_aggressive_reg, ransac_reg, huber_reg,
                      hist_gradient_boosting_reg, tweedie_reg, poisson_reg,
                      gaussian_process_reg, voting_reg, theil_sen_reg,
                      extra_trees_reg]

# Training and calculating R2 for each model
model_names = ["Linear Regression", "Ridge Regression", "Lasso Regression",
               "Elastic Net Regression", "SVR", "NuSVR", "Linear SVR",
               "Decision Tree Regression", "Random Forest Regression",
               "Gradient Boosting Regression", "AdaBoost Regression",
               "Bagging Regression", "K-Nearest Neighbors Regression",
               "Kernel Ridge Regression",
               "MLP Regression", "Passive Aggressive Regression", "RANSAC Regression",
               "Huber Regression", "Histogram-based GBR",
               "Tweedie Regression", "Poisson Regression",
                "Gaussian Process Regression",
                "Voting Regression", "Theil-Sen Regression", "Extra Trees Regression"]


def regular_regression_models(X, y, num_folds=5):

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    r2_scores = []
    r2_scores_train = []

    # Iterate through each fold
    for train_index, test_index in kf.split(X):  # Replace X with your data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # Replace X with your data
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # Replace y with your target variable

        r2_scores_method = []
        r2_scores_train_method = []
        for model_name, model in zip(model_names, regression_models):
            model.fit(X_train, y_train)
            r2_scores_method.append(r2_score(y_test, model.predict(X_test)))

            r2_scores_train_method.append(r2_score(y_train, model.predict(X_train)))

        r2_scores.append(r2_scores_method)
        r2_scores_train.append(r2_scores_train_method)

    r2_scores = pd.DataFrame(np.array(r2_scores).T, index=model_names)
    r2_scores_train = pd.DataFrame(np.array(r2_scores_train).T, index=model_names)

    return r2_scores, r2_scores_train


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


def tf_regression_model_with_probability(input_data_size, input_shape):

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
    return model
