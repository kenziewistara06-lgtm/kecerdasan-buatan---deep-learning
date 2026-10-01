# Example 4.19 SimpleRNN2.py

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, Input
from keras import optimizers


# Membuat model
model = Sequential()

model.add(Input(shape=(49, 1)))
model.add(SimpleRNN(50, return_sequences=False))
model.add(Dense(46))
model.add(Activation('softmax'))


# Optimizer
adam = optimizers.Adam(learning_rate=0.001)


# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=adam,
    metrics=['accuracy']
)


# Menampilkan ringkasan model
print(model.summary())