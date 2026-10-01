# Exercise 4.12
# Webcam Classification using DenseNet

import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import (
    vgg16,
    resnet50,
    mobilenet,
    inception_v3,
    densenet
)

# Initialize the model

# model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')
model = densenet.DenseNet121(weights='imagenet')

print(model.summary())

# Open webcam
camera = cv2.VideoCapture(0)

image_size = 224
# image_size = 299  # Use this for InceptionV3

while camera.isOpened():
    ok, cam_frame = camera.read()

    if not ok:
        print("Camera tidak dapat dibaca.")
        break

    # Resize frame
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # Convert image to array
    numpy_image = img_to_array(frame)

    # Add batch dimension
    image_batch = np.expand_dims(numpy_image, axis=0)

    # Preprocess image
    # processed_image = vgg16.preprocess_input(image_batch.copy())
    # processed_image = resnet50.preprocess_input(image_batch.copy())
    # processed_image = mobilenet.preprocess_input(image_batch.copy())
    # processed_image = inception_v3.preprocess_input(image_batch.copy())
    processed_image = densenet.preprocess_input(image_batch.copy())

    # Get predicted probabilities
    predictions = model.predict(processed_image, verbose=0)

    # Decode prediction
    label = decode_predictions(predictions, top=1)

    # Display prediction result
    cv2.putText(
        cam_frame,
        "DenseNet: {}, {:.1f}%".format(
            label[0][0][1],
            label[0][0][2] * 100
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Display webcam
    cv2.imshow('DenseNet Webcam Classification', cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break

# Release camera
camera.release()
cv2.destroyAllWindows()