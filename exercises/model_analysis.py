import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from exercises.regression_models import *
from exercises.data_preparation import load_complete_data, load_data
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
sns.set_context('talk')
props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
prop_ind = 1
X, y, Z = load_complete_data()

y = y[props[prop_ind]]
y_max = y.max()
y = y/y_max


X = pd.concat([X, Z], axis=1)
r2_scores, r2_scores_train = regular_regression_models(X, y)
r2_scores[r2_scores < 0] = 0
r2_scores = r2_scores[r2_scores.sum(axis=1) > 0]

r2_scores_train[r2_scores_train < 0] = 0
sns.boxplot(r2_scores_train.T)
plt.xticks(rotation=90)
plt.title('Training R2 for ' + props[prop_ind])
plt.savefig(f'figs/Regula_ML_train_{props[prop_ind]}.png', bbox_inches='tight')
plt.show()

sns.boxplot(r2_scores.T)
plt.xticks(rotation=90)
plt.title('Testing R2 for ' + props[prop_ind])
plt.savefig(f'figs/Regula_ML_test_{props[prop_ind]}.png', bbox_inches='tight')
plt.show()

print(f'Max R2 for {props[prop_ind]} is {r2_scores.max().max()}')



# X = Z_scaled.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

input_shape = X.shape[1:]
input_data_size = X.shape[0]

model = tf_regression_model(input_shape=input_shape)
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))


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