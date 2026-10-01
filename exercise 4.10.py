# Exercise 4.10
# Image Classification using VGG19

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

# Load VGG19 model
model = VGG19(weights='imagenet')

# Load image
img_path = 'elephant.jpg'
img = load_img(img_path, target_size=(224, 224))

# Convert image to array
x = img_to_array(img)

# Add batch dimension
x = np.expand_dims(x, axis=0)

# Preprocess image
x = preprocess_input(x)

# Make prediction
predictions = model.predict(x)

# Display top 5 predictions
print('Predicted:')
for i, prediction in enumerate(decode_predictions(predictions, top=5)[0]):
    print(i + 1, prediction)