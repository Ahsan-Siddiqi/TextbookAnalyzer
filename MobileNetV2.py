import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the MobileNetV2 model pre-trained on ImageNet
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Resize the frame to match the input size of the model (224x224 for MobileNetV2)
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Preprocess the frame for MobileNetV2 (normalize the input image)
    preprocessed_frame = preprocess_input(np.expand_dims(resized_frame, axis=0))

    # Perform prediction
    predictions = model.predict(preprocessed_frame)
    
    # Decode the prediction (get the top 3 predictions)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Get the highest confidence prediction
    label, confidence = decoded_predictions[0][1], decoded_predictions[0][2]

    # Display the result on the frame
    text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Textbook Detection', frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
