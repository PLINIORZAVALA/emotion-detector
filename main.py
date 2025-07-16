# main.py
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time
import matplotlib.pyplot as plt

# Visualización de emociones
x = [0,1,2,3,4,5,6]
y = [0.0] * 7
plt.ion()
figura1 = plt.figure()
my_colors = 'rgbykmc'

# Clases de emociones
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Carga de modelos
faceNet = cv2.dnn.readNet('face_detector/deploy.prototxt',
                          'face_detector/res10_300x300_ssd_iter_140000.caffemodel')
emotionModel = load_model('model/modelFEC.h5')

# Función para predecir emociones
def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces, locs, preds = [], [], []

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0]] * 2)
            (Xi, Yi, Xf, Yf) = box.astype("int")
            Xi, Yi = max(0, Xi), max(0, Yi)
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)
            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))
            pred = emotionModel.predict(face2, verbose=0)
            preds.append(pred[0])
    return (locs, preds)

# Captura en vivo
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time_actualframe = time_prevframe = 0

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=640)
    locs, preds = predict_emotion(frame, faceNet, emotionModel)

    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        label = "{}: {:.0f}%".format(classes[np.argmax(pred)], np.max(pred) * 100)

        cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi + 5, Yi - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

        plt.clf()
        y = pred
        plt.xticks(x, classes)
        plt.grid(True)
        plt.ylim([0.0, 1.0])
        plt.bar(x, y, color=my_colors, width=1)
        figura1.canvas.draw()

    time_actualframe = time.time()
    if time_actualframe > time_prevframe:
        fps = 1 / (time_actualframe - time_prevframe)
    time_prevframe = time_actualframe
    cv2.putText(frame, f"{int(fps)} FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
