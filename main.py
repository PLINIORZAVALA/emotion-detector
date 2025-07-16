import cv2
import imutils
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json  # Change here
from tensorflow.keras.preprocessing.image import img_to_array

# Emotion classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
colors = 'rgbykmc'
x = list(range(len(classes)))
y = [0.0] * len(classes)

# Enable interactive mode for real-time updates
plt.ion()
figure = plt.figure()

# Load face detector model (Caffe)
face_model_path = "face_detector"
prototxtPath = f"{face_model_path}/deploy.prototxt"
weightsPath = f"{face_model_path}/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load emotion classification model from JSON + H5
with open("model/modelFEC.json", "r") as json_file:
    model_json = json_file.read()

emotionModel = model_from_json(model_json)
emotionModel.load_weights("model/modelFEC.h5")

# Function to detect faces and predict emotion
def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([
                frame.shape[1], frame.shape[0],
                frame.shape[1], frame.shape[0]
            ])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            Xi, Yi = max(0, Xi), max(0, Yi)

            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face_array = img_to_array(face)
            face_array = np.expand_dims(face_array, axis=0)

            faces.append(face_array)
            locs.append((Xi, Yi, Xf, Yf))
            pred = emotionModel.predict(face_array)
            preds.append(pred[0])

    return (locs, preds)

# Initialize camera
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# FPS calculation
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    locs, preds = predict_emotion(frame, faceNet, emotionModel)

    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        (angry, disgust, fear, happy, neutral, sad, surprise) = pred

        label = f"{classes[np.argmax(pred)]}: {np.max(pred) * 100:.0f}%"

        # Draw label and box
        cv2.rectangle(frame, (Xi, Yi - 40), (Xf, Yi), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi + 5, Yi - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 2)

        # Update live emotion bar chart
        y = [angry, disgust, fear, happy, neutral, sad, surprise]
        plt.clf()
        plt.xticks(x, classes)
        plt.grid(True)
        plt.ylim([0.0, 1.0])
        plt.bar(x, y, color=colors, width=0.8)
        figure.canvas.draw()

    # FPS display
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    cv2.putText(frame, f"{int(fps)} FPS", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
