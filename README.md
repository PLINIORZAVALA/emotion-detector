# Real-Time Emotion Detection using CNN and OpenCV

This project implements a real-time facial emotion detection system using a convolutional neural network (CNN) model trained on the FER2013 dataset, combined with face detection using OpenCV and dynamic emotion visualization with Matplotlib.

---

## Project Structure

```

emotion-detector/
│
├── face\_detector/                          # Pre-trained face detection model (Caffe)
│   ├── deploy.prototxt                     # Model architecture definition
│   ├── res10\_300x300\_ssd\_iter\_140000.caffemodel  # Model weights
│   └── z\_links.txt                         # Optional: download links or reference
│
├── model/                                  # Emotion classification model
│   ├── 67emotion\_human.h5                  # CNN model weights
│   └── 67emotion\_human.json                # CNN model architecture
│
├── main.py                                 # Main script for real-time emotion detection
│
├── requirements.txt                        # Python dependencies
│
└── README.md                               # Documentation and instructions

````

---

## Features

- Real-time face detection using OpenCV with SSD ResNet model.
- Facial emotion classification into 7 classes:  
  `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- Real-time emotion probabilities visualized in a dynamic Matplotlib bar chart.
- Display of the dominant emotion with percentage and current FPS on-screen.

---

## Requirements

- Python 3.12.3 or higher  
- Webcam (built-in or USB)  
- Recommended: TensorFlow CPU/GPU and Matplotlib

---

## Installation and Running

### 1. Clone the repository

```bash
git clone https://github.com/your-user/emotion-detector.git
cd emotion-detector
````

### 2. Create a virtual environment

```bash
python3 -m venv venv310
```

### 3. Activate the virtual environment

* On **Linux/macOS**:

```bash
source venv310/bin/activate
```

* On **Windows (PowerShell)**:

```bash
venv310\Scripts\activate
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

> 💡 If you have a CUDA-capable GPU, consider installing `tensorflow-gpu` for better performance.

### 5. Run the application

```bash
python main.py
```

Press `q` in the OpenCV window to stop the camera and close the program.

---

## Models Used

* **Face detector**: Pre-trained `ResNet SSD` (Single Shot Multibox Detector) from OpenCV using the Caffe framework.
* **Emotion classifier**: Custom CNN trained on the FER2013 dataset, loaded from JSON and `.h5` weights.

---

## Visualization

* A **bounding box** with the predicted emotion and its confidence is shown around each detected face.
* A **bar chart** is updated in real time, showing the probability for each of the 7 emotions.

---

## License

This project is distributed under the MIT License. You are free to use, modify, and share it for educational or research purposes.

---

## Credits

* Original idea based on a project by [David Revelo Luna](https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg)
* Enhanced and adapted by **Plinior Zavala** with improved visualization, modular structure, and compatibility for both Windows and Linux environments.

