# Exercise 4.14
# Select Classification Model Using If-Else

from keras.applications.imagenet_utils import decode_predictions
from classification_models.keras import Classifiers

import numpy as np
import cv2


# Display model options
print("===================================")
print("     MODEL CLASSIFICATION")
print("===================================")
print("1. VGG16")
print("2. ResNet50")
print("3. MobileNetV2")
print("4. DenseNet201")
print("5. InceptionV3")
print("===================================")

choice = input("Pilih model (1-5): ")


# Select model using if-elif-else

if choice == '1':

    clf, preprocess_input = Classifiers.get('vgg16')
    model_name = 'VGG16'
    sz = 224

elif choice == '2':

    clf, preprocess_input = Classifiers.get('resnet50')
    model_name = 'ResNet50'
    sz = 224

elif choice == '3':

    clf, preprocess_input = Classifiers.get('mobilenetv2')
    model_name = 'MobileNetV2'
    sz = 224

elif choice == '4':

    clf, preprocess_input = Classifiers.get('densenet201')
    model_name = 'DenseNet201'
    sz = 224

elif choice == '5':

    clf, preprocess_input = Classifiers.get('inceptionv3')
    model_name = 'InceptionV3'
    sz = 299

else:

    print("Pilihan model tidak valid.")
    exit()


print("\nModel yang dipilih:", model_name)


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

    # Display webcam
    cv2.imshow("Classification", cam_frame)

    # Press ESC to quit
    key = cv2.waitKey(30)

    if key == 27:
        break


# Release camera
camera.release()
cv2.destroyAllWindows()