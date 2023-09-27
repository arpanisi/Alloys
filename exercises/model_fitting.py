from exercises.data_preparation import load_complete_data, load_data
from exercises.GAN import generate_synthetic_data
from exercises.regression_models import tf_regression_model, regular_regression_models
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

props = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
prop_ind = 0
X, y, Z = load_complete_data()


synthetic_alloys, synthetic_props, synthetic_Z = generate_synthetic_data(number_of_alloys=1000)

X_total = pd.concat([X, synthetic_alloys], axis=0)
Z_total = pd.concat([Z, synthetic_Z], axis=0)
y_total = pd.concat([y, synthetic_props], axis=0)

X_tot = pd.concat([X_total, Z_total], axis=1)

y_prop = y_total[props[prop_ind]]
y_max = y_prop.max()
y_prop = y_prop/y_max

r2_scores, r2_scores_train = regular_regression_models(X_total, y_prop)

r2_scores[r2_scores < 0] = 0
r2_scores = r2_scores[r2_scores.sum(axis=1) > 0]

r2_scores_train[r2_scores_train < 0] = 0
print(f'Max R2 for {props[prop_ind]} is {r2_scores.max().max()}')


X_train, X_test, y_train, y_test = train_test_split(X_total, y_prop,
                                                    test_size=0.3, random_state=42)

input_shape = X_train.shape[1:]
model = tf_regression_model(input_shape)
history = model.fit(X_train, y_train,
                    epochs=200, validation_data=(X_test, y_test))

print('Training R2:', r2_score(y_train, model.predict(X_train)))
print('Testing R2:', r2_score(y_test, model.predict(X_test)))

plt.plot(history.epoch, history.history['loss'], c='k', label='Training')
plt.plot(history.epoch, history.history['val_loss'], c='r', label='Testing')
plt.title(props[prop_ind])
plt.legend()
plt.show()

plt.savefig('Loss_curve', bbox_inches='tight')