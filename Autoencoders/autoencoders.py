import random
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt


# loads the popular "mnist" training dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1
x_train = x_train/255.0
# scales the data. pixel values range from 0 to 255, so this makes it range 0 to 1
x_test = x_test/255.0


encoder_input = keras.Input(shape=(28, 28), name='img')
x = keras.layers.Flatten()(encoder_input)
encoder_output = keras.layers.Dense(64, activation="relu")(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')

decoder_input = keras.layers.Dense(64, activation="relu")(encoder_output)
x = keras.layers.Dense(784, activation="relu")(decoder_input)
decoder_output = keras.layers.Reshape((28, 28))(x)

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()

autoencoder.compile(opt, loss='mse')

epochs = 3

for epoch in range(epochs):

    history = autoencoder.fit(
        x_train,
        x_train,
        epochs=1,
        batch_size=32, validation_split=0.10
    )
    # autoencoder.save(f"models/AE-{epoch+1}.model")

for d in x_test[:30]:  # just show 5 examples, feel free to show all or however many you want!

    ae_out = autoencoder.predict([d.reshape(-1, 28, 28)])
    img = ae_out[0]

    cv2.imshow("decoded", img)
    cv2.imshow("original", np.array(d))
    cv2.waitKey(1000)  # wait 1000ms, 1 second, and then show the next.


def add_noise(img, random_chance=5):
    noisy = []
    for row in img:
        new_row = []
        for pix in row:
            if random.choice(range(100)) <= random_chance:
                new_val = random.uniform(0, 1)
                new_row.append(new_val)
            else:
                new_row.append(pix)
        noisy.append(new_row)
    return np.array(noisy)


def remove_values(img, random_chance=5):
    noisy = []
    for row in img:
        new_row = []
        for pix in row:
            if random.choice(range(100)) <= random_chance:
                new_val = 0  # changing this to be 0
                new_row.append(new_val)
            else:
                new_row.append(pix)
        noisy.append(new_row)
    return np.array(noisy)


# slightly higher chance so we see more impact
some_hidden = remove_values(x_train[0], random_chance=15)
plt.imshow(some_hidden, cmap="gray")
plt.show()

ae_out = autoencoder.predict([some_hidden.reshape(-1, 28, 28)])
# predict is done on a vector, and returns a vector, even if its just 1 element, so we still need to grab the 0th
img = ae_out[0]
plt.imshow(ae_out[0], cmap="gray")
plt.show()

# slightly higher chance so we see more impact
some_hidden = remove_values(x_train[0], random_chance=35)
plt.imshow(some_hidden, cmap="gray")
plt.show()

ae_out = autoencoder.predict([some_hidden.reshape(-1, 28, 28)])
# predict is done on a vector, and returns a vector, even if its just 1 element, so we still need to grab the 0th
img = ae_out[0]
plt.imshow(ae_out[0], cmap="gray")
plt.show()