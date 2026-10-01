# Exercise 4.11
# VGG19 Webcam Classifier

import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import vgg19
import numpy as np

image_size = 224

# Load VGG19 model
model = vgg19.VGG19(weights='imagenet')
print(model.summary())

# Open webcam
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ok, cam_frame = camera.read()

    if not ok:
        print("Camera tidak dapat dibaca.")
        break

    # Resize webcam frame
    frame = cv2.resize(cam_frame, (image_size, image_size))

    # Convert image to array
    numpy_image = img_to_array(frame)

    # Add batch dimension
    image_batch = np.expand_dims(numpy_image, axis=0)

    # Preprocess image
    processed_image = vgg19.preprocess_input(image_batch.copy())

    # Get predicted probabilities
    predictions = model.predict(processed_image, verbose=0)

    # Decode prediction
    label = decode_predictions(predictions, top=1)

    # Display prediction result
    cv2.putText(
        cam_frame,
        "VGG19: {}, {:.1f}%".format(
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
    cv2.imshow('VGG19 Webcam Classifier', cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()