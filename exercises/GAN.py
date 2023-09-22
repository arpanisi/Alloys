from exercises.data_preparation import load_complete_data, load_data
from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
import numpy as np
import pandas as pd


def make_generator_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(input_shape[0], activation='relu')])

    return model

def make_discriminator_model(input_shape):

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1)])

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def generate_synthetic_data(number_of_alloys=1000, prop='tot', epochs=2000):

    if prop == 'tot':
        prop = ['YieldStr(MPa)', 'Ductility (%)', 'Hardness (HV)']
        X, y, Z = load_complete_data()
    else:
        X, y, Z = load_data(col=prop)
    elem_col = X.columns
    other_cols = Z.columns

    y_max = y.max()
    y = y/y_max

    Z_max = Z.max()
    Z_scaled = Z/Z_max

    X = pd.concat([X, y, Z_scaled], axis=1)
    input_shape = X.shape[1:]

    generator = make_generator_model(input_shape)
    discriminator = make_discriminator_model(input_shape)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    data = X.values.copy()

    for epoch in range(epochs):
        noise = np.random.random(data.shape)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(noise, training=True)

            real_output = discriminator(data, training=True)
            fake_output = discriminator(generated_data, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')


    generated_noise = np.random.random((number_of_alloys, input_shape[0]))
    generated_data = pd.DataFrame(generator.predict(generated_noise), columns=X.columns)

    generated_alloys = generated_data[elem_col]
    generated_alloys = generated_alloys.div(generated_alloys.sum(axis=1), axis=0)
    generated_props = generated_data[prop] * y_max
    generated_Z = generated_data[other_cols] * Z_max

    return generated_alloys, generated_props, generated_Z