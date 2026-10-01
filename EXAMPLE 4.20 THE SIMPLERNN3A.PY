# Example 4.20
# Modified based on:
# https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input


# ============================================================
# 2. CONVERT DATA INTO DATASET MATRIX
# ============================================================

def convertToMatrix(data, step):
    X, Y = [], []

    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])

    return np.array(X), np.array(Y)


# ============================================================
# 3. LOAD AIR PASSENGERS DATASET
# ============================================================

# Dataset source:
# https://www.kaggle.com/rakannimer/air-passengers

df = pd.read_csv(
    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
    usecols=[1],
    engine='python'
)

print(df.head())


# ============================================================
# 4. VISUALIZE THE DATASET
# ============================================================

plt.plot(df)
plt.title('Air Passengers Dataset')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.show()


# ============================================================
# 5. SET STEP AND TRAINING DATA SIZE
# ============================================================

step = 4

N = df.shape[0]

Tp = int(df.shape[0] * 0.8)


# ============================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

values = df.values

train = values[0:Tp, :]
test = values[Tp:N, :]


# ============================================================
# 7. ADD STEP ELEMENTS INTO TRAIN AND TEST
# ============================================================

test = np.append(
    test,
    np.repeat(test[-1, ], step)
)

train = np.append(
    train,
    np.repeat(train[-1, ], step)
)


# ============================================================
# 8. CONVERT DATA INTO MATRIX
# ============================================================

trainX, trainY = convertToMatrix(
    train,
    step
)

testX, testY = convertToMatrix(
    test,
    step
)


# ============================================================
# 9. RESHAPE DATA FOR SIMPLE RNN
# ============================================================

trainX = np.reshape(
    trainX,
    (trainX.shape[0], 1, trainX.shape[1])
)

testX = np.reshape(
    testX,
    (testX.shape[0], 1, testX.shape[1])
)


# ============================================================
# 10. BUILD SIMPLE RNN MODEL
# ============================================================

model = Sequential()

model.add(
    Input(shape=(1, step))
)

model.add(
    SimpleRNN(
        units=32,
        activation='relu'
    )
)

model.add(
    Dense(
        8,
        activation='relu'
    )
)

model.add(
    Dense(1)
)


# ============================================================
# 11. COMPILE MODEL
# ============================================================

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop'
)


# ============================================================
# 12. DISPLAY MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# 13. TRAIN MODEL
# ============================================================

model.fit(
    trainX,
    trainY,
    epochs=10,
    batch_size=16,
    verbose=2
)


# ============================================================
# 14. MAKE PREDICTIONS
# ============================================================

trainPredict = model.predict(
    trainX,
    verbose=0
)

testPredict = model.predict(
    testX,
    verbose=0
)


# Combine training and testing predictions
predicted = np.concatenate(
    (trainPredict, testPredict),
    axis=0
)


# ============================================================
# 15. EVALUATE MODEL
# ============================================================

trainScore = model.evaluate(
    trainX,
    trainY,
    verbose=0
)

print("Training Loss:", trainScore)


# ============================================================
# 16. VISUALIZE ACTUAL DATA AND PREDICTION
# ============================================================

index = df.index.values

plt.plot(
    df,
    label='Data'
)

plt.plot(
    predicted,
    label='Prediction'
)

plt.axvline(
    df.index[Tp],
    color='red'
)

plt.legend()
plt.title('Air Passengers Prediction')
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.show()