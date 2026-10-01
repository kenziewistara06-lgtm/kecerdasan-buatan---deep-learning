# Exercise 4.13
# Compare Different Classification Models

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

import numpy as np
import cv2


# Select model
# VGG16
clf, preprocess_input = Classifiers.get('vgg16')
model_name = 'VGG16'
sz = 224

# ResNet50
# clf, preprocess_input = Classifiers.get('resnet50')
# model_name = 'ResNet50'
# sz = 224

# MobileNetV2
# clf, preprocess_input = Classifiers.get('mobilenetv2')
# model_name = 'MobileNetV2'
# sz = 224

# DenseNet201
# clf, preprocess_input = Classifiers.get('densenet201')
# model_name = 'DenseNet201'
# sz = 224

# InceptionV3
# clf, preprocess_input = Classifiers.get('inceptionv3')
# model_name = 'InceptionV3'
# sz = 299


# Create model
model = clf(
    input_shape=(sz, sz, 3),
    weights='imagenet',
    classes=1000
)

model.summary()

# Open webcam
camera = cv2.VideoCapture(0)

while True:

    ret, cam_frame = camera.read()

    if not ret:
        print("Camera tidak dapat dibaca.")
        break

    # Resize frame
    frame = cv2.resize(cam_frame, (sz, sz))

    # Convert image to array
    image = np.asarray(frame)

    # Add batch dimension
    image = np.expand_dims(image, 0)

    # Preprocess image
    image = preprocess_input(image)

    # Prediction
    preds = model.predict(image, verbose=0)

    # Decode prediction
    label = decode_predictions(preds, top=1)

    # Display prediction
    text = "{}: {}, {:.1f}%".format(
        model_name,
        label[0][0][1],
        label[0][0][2] * 100
    )

    cv2.putText(
        cam_frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Classification", cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break


camera.release()
cv2.destroyAllWindows()