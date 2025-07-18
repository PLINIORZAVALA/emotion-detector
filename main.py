import cv2
import imutils
import numpy as np
import time
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf

# ========== Clase BlurLayer EXACTA DEL ENTRENAMIENTO ==========
@tf.keras.utils.register_keras_serializable()
class BlurLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BlurLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        blur_kernel = tf.constant([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]], dtype=tf.float32)
        blur_kernel = blur_kernel / tf.reduce_sum(blur_kernel)
        blur_kernel = tf.reshape(blur_kernel, [3, 3, 1, 1])
        blur_kernel = tf.tile(blur_kernel, [1, 1, channels, 1])
        self.blur_filter = tf.Variable(initial_value=blur_kernel, trainable=False, name='blur_filter')

    def call(self, x):
        return tf.nn.depthwise_conv2d(x, self.blur_filter, strides=[1, 1, 1, 1], padding='SAME')

# Registrar clases personalizadas
get_custom_objects()['BlurLayer'] = BlurLayer
get_custom_objects()['Sequential'] = Sequential

# ========== Clases de emoción ==========
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
colors = 'rgbykmc'
x = list(range(len(classes)))
y = [0.0] * len(classes)

# ========== Graficado en vivo ==========
plt.ion()
figure = plt.figure()

# ========== Detector de rostros (modelo Caffe) ==========
face_model_path = "face_detector"
prototxtPath = f"{face_model_path}/deploy.prototxt"
weightsPath = f"{face_model_path}/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# ========== Cargar modelo desde JSON + pesos ==========
with open("model/67emotion_human.json", "r") as json_file:
    model_json = json_file.read()

emotionModel = model_from_json(model_json, custom_objects={
    "BlurLayer": BlurLayer,
    "Sequential": Sequential
})

emotionModel.load_weights("model/67emotion_human.h5")
print("✅ Modelo cargado exitosamente")

# ========== Función de predicción ==========
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
            face_array = face_array / 255.0

            faces.append(face_array)
            locs.append((Xi, Yi, Xf, Yf))
            pred = emotionModel.predict(face_array, verbose=0)
            preds.append(pred[0])

    return (locs, preds)

# ========== Captura de cámara ==========
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

        # Dibujar etiqueta y caja
        cv2.rectangle(frame, (Xi, Yi - 40), (Xf, Yi), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi + 5, Yi - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 2)

        # Actualizar gráfica de emociones
        y = [angry, disgust, fear, happy, neutral, sad, surprise]
        plt.clf()
        plt.xticks(x, classes)
        plt.grid(True)
        plt.ylim([0.0, 1.0])
        plt.bar(x, y, color=colors, width=0.8)
        figure.canvas.draw()

    # Mostrar FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
    prev_frame_time = new_frame_time

    cv2.putText(frame, f"{int(fps)} FPS", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ========== Liberar recursos ==========
cam.release()
cv2.destroyAllWindows()
