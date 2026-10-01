# Exercise 4.9
# VGG19 Model

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np


# Load VGG19 model
model = VGG19(weights='imagenet')


# Display model summary
print(model.summary())