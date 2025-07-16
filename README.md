# Real-Time Emotion Detection using CNN and OpenCV

This project implements a real-time facial emotion detection system using a convolutional neural network (CNN) model trained on the FER2013 dataset, combined with face detection using OpenCV and dynamic emotion visualization with Matplotlib.

---

## Project Structure

```

emotion-detector/
│
├── face\_detector/ # Contains the pre-trained model for face detection (based on Caffe)
│ ├── deploy.prototxt # Model architecture definition
│ └── res10\_300x300\_ssd\_iter\_140000.caffemodel # Face detection model weights
│
├── model/
│ └── modelFEC.h5 # CNN model trained to classify facial emotions
│
├── main.py # Main script that runs emotion detection in real time
│
├── requirements.txt # List of Virtual environment dependencies (required libraries)
│
└── README.md # Project description and documentation

````

---

## 🔧 Features

- Real-time face detection with OpenCV and SSD model.
- Facial emotion classification: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- Visualization of probabilities in real-time bar charts with Matplotlib.
- Display of the dominant emotion on the detected face, along with the performance in FPS.

---

## Requirements

- Python 3.7 or higher
- Working webcam

---

## Installation and Running

### 1. Clone the repository

```bash
git clone https://github.com/your-user/emotion-detector.git
cd emotion-detector
````

### 2. Create a virtual environment with `venv`

```bash
python -m venv venv
```

### 3. Activate the virtual environment

* On **Windows**:

```bash
venv\Scripts\activate
```

* On **Linux/macOS**:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: If your computer has a CUDA-compatible GPU, you can install the optimized version of TensorFlow (`tensorflow-gpu`).

### 5. Run the system

```bash
python main.py
```

Press `q` to close the camera window and end the program.

---

## Models used

* **Face detector**: Based on the pre-trained `ResNet SSD` (Single Shot Multibox Detector) OpenCV model with Caffe architecture.
* **Emotion detector**: Convolutional neural network trained on the FER2013 dataset.

---

## Visualization

During execution:

* A **bounding box** is displayed with the dominant emotion and its probability on the detected face.
* A **real-time** bar chart is displayed with the probability of each emotion.

---

## License

This project is distributed under the MIT License. You may use, modify, and redistribute it for educational and research purposes.

---

## Credits

* Original project based on the YouTube channel [David Revelo Luna](https://www.youtube.com/channel/UCr_dJOULDvSXMHA1PSHy2rg)
* Modified and extended by Plinior Zavala with emotion visualization and a professional project structure.

---

```