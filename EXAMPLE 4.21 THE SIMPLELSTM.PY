# Example 4.21 SimpleLSTM.py
# Based on:
# https://www.tensorflow.org/guide/keras/rnn


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 2. CREATE LSTM MODEL
# ============================================================

model = keras.Sequential([
    layers.Input(shape=(10,)),

    layers.Embedding(
        input_dim=100,
        output_dim=32
    ),

    layers.LSTM(64),

    layers.Dense(1)
])


# ============================================================
# 3. DISPLAY MODEL SUMMARY
# ============================================================

model.summary()