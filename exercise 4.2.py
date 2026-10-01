# Exercise 4.2
# Multiple Layer Perceptron with Three Inputs

from sklearn.neural_network import MLPClassifier

# Three inputs
X = [
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [0., 1., 1.],
    [1., 0., 0.],
    [1., 0., 1.],
    [1., 1., 0.],
    [1., 1., 1.]
]

# Output for logical OR
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(
    solver='lbfgs',
    alpha=1e-5,
    hidden_layer_sizes=(5, 2),
    random_state=1
)

# Train the model
clf.fit(X, y)

# Test the model
print(clf.predict([
    [0., 0., 0.],
    [1., 1., 1.],
    [1., 0., 0.]
]))

# Display the shape of weights
print([coef.shape for coef in clf.coefs_])
