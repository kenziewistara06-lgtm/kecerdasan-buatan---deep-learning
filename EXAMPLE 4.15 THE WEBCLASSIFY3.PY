# Example 4.15
# Image Classification using classification_models

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

import numpy as np
import cv2


# Set up a model
clf, preprocess_input = Classifiers.get('vgg16')

# clf, preprocess_input = Classifiers.get('resnet50')
# clf, preprocess_input = Classifiers.get('mobilenetv2')
# clf, preprocess_input = Classifiers.get('densenet201')

sz = 224

# clf, preprocess_input = Classifiers.get('inceptionv3')
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

    # Read frame from webcam
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

    # Predict
    preds = model.predict(image, verbose=0)

    # Decode prediction
    label = decode_predictions(preds, top=1)

    # Display prediction
    cv2.putText(
        cam_frame,
        "{}, {:.1f}%".format(
            label[0][0][1],
            label[0][0][2] * 100
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow("Classification", cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break


# Release camera
camera.release()
cv2.destroyAllWindows()