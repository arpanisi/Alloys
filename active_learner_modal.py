from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from exercises.data_preparation import load_data
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
import numpy as np
from sklearn.metrics import r2_score
props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
prop_ind = 0
X, y, _ = load_data(col=props[prop_ind])


X = np.random.choice(np.linspace(0, 20, 10000), size=(200, 2), replace=False).reshape(-1, 1)
y = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)
def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
         + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))


regressor = ActiveLearner(
    estimator=GaussianProcessRegressor(kernel=kernel),
    query_strategy=GP_regression_std,
    X_training=X, y_training=y
)

query_strategy = uncertainty_sampling(regressor)

query_idx, query_instance = regressor.query(X)

n_queries = 10
for idx in range(n_queries):
    query_idx, query_instance = regressor.query(X)
    regressor.teach(X[query_idx].reshape(1, -1), y[que])