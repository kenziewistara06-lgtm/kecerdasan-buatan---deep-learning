# Exercise 4.3
# Multiple Layer Perceptron Classification
# Using 5 neurons for the Dense layer

from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time


# Load dataset
whole_data = load_breast_cancer()

X_data = whole_data.data
y_data = whole_data.target


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.3,
    random_state=7
)


features = X_train.shape[1]


# MLP 3-layer Model
model = Sequential()

model.add(
    Dense(
        5,
        activation='relu',
        input_shape=[X_train.shape[1]]
    )
)

model.add(Dense(5, activation='relu'))
model.add(Dense(1))


# Display model information
print(model.summary())


# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mse']
)


# Start training time
start_time = time.time()

model.fit(
    X_train,
    y_train,
    batch_size=50,
    validation_split=0.2,
    epochs=100,
    verbose=1
)

# Calculate training time
training_time = time.time() - start_time


# Evaluate model
results = model.evaluate(X_test, y_test, verbose=0)

print("\nTraining time:", training_time, "seconds")
print("Loss:", results[0])
print("MSE:", results[1])