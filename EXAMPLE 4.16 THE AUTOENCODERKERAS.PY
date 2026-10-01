# Example 4.16 AutoencoderKeras.py
# Modified from:
# https://stackabuse.com/autoencoders-for-image-reconstruction-in-python-and-keras/

from keras.datasets import mnist
import numpy as np

# =========================
# Load Dataset MNIST
# =========================
(x_train, _), (x_test, _) = mnist.load_data()

# Normalisasi nilai pixel menjadi 0-1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# =========================
# Import Keras Layers
# =========================
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model


# =========================
# Build Autoencoder
# =========================
def build_autoencoder(img_shape, code_size):

    # Encoder
    encoder = Sequential()
    encoder.add(InputLayer(shape=img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # Decoder
    decoder = Sequential()
    decoder.add(InputLayer(shape=(code_size,)))

    # int() digunakan agar Keras menerima nilai integer Python
    decoder.add(Dense(int(np.prod(img_shape))))

    decoder.add(Reshape(img_shape))

    return encoder, decoder


# Ukuran gambar
IMG_SHAPE = x_train[0].shape

# Ukuran code/latent representation
encoder, decoder = build_autoencoder(IMG_SHAPE, 32)


# =========================
# Connect Encoder + Decoder
# =========================
inp = Input(shape=IMG_SHAPE)

code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)


# =========================
# Compile Model
# =========================
autoencoder.compile(
    optimizer='adamax',
    loss='mse'
)

print(autoencoder.summary())


# =========================
# Train Autoencoder
# =========================
history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    validation_data=(x_test, x_test)
)


# =========================
# Plot Training Loss
# =========================
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()


# =========================
# Display Image
# =========================
def show_image(x):
    plt.imshow(
        np.clip(x + 0.5, 0, 1),
        cmap='gray'
    )
    plt.axis('off')


# =========================
# Visualize Reconstruction
# =========================
def visualize(img, encoder, decoder):

    # Encode image
    code = encoder.predict(
        img[None],
        verbose=0
    )[0]

    # Decode image
    reco = decoder.predict(
        code[None],
        verbose=0
    )[0]

    # Original image
    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    # Encoded representation
    plt.subplot(1, 3, 2)
    plt.title("Code")

    plt.imshow(
        code.reshape([code.shape[-1] // 2, -1]),
        cmap='gray'
    )

    plt.axis('off')

    # Reconstructed image
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reco)

    plt.tight_layout()
    plt.show()


# =========================
# Display 5 Test Images
# =========================
for i in range(5):

    img = x_test[i]

    visualize(
        img,
        encoder,
        decoder
    )