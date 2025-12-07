# 🤟 ASL Sign Language Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-brightgreen.svg)
![Real-time](https://img.shields.io/badge/Real--time-Detection-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**Real-time American Sign Language (ASL) detection and text-to-speech conversion using Computer Vision and Machine Learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Demo](#-demo) • [Architecture](#-architecture)

</div>

---

## 📖 Overview

ASL Sign Language Detection System is an intelligent computer vision application that recognizes **American Sign Language (ASL)** hand gestures in real-time. Using **MediaPipe** for hand tracking and **Random Forest** classifier for gesture recognition, it converts sign language into text and speech, making communication more accessible for the deaf and hard-of-hearing community.

### 🎯 Key Highlights

- **Real-time Detection**: Instant ASL gesture recognition via webcam
- **High Accuracy**: 95%+ accuracy with Random Forest classifier
- **Text-to-Speech**: Automatic speech synthesis using pyttsx3
- **Word Building**: Character confirmation and word construction
- **Special Commands**: Space, Delete, and Nothing gestures
- **Hand Tracking**: 21-point hand landmark detection with MediaPipe
- **Custom Dataset**: Trained on ASL alphabet dataset
- **Production Ready**: Complete pipeline from data extraction to deployment

---

## ✨ Features

### 🤖 **Intelligent Detection System**

#### Core Capabilities:
- **Hand Landmark Detection**: 21 keypoints per hand
- **Gesture Recognition**: Full ASL alphabet (A-Z)
- **Real-time Processing**: ~30 FPS on standard webcam
- **Confirmation System**: 15-frame confirmation to prevent false positives
- **Word Construction**: Automatic character concatenation

### 🎤 **Text-to-Speech Integration**
- **Voice Output**: Press 'S' to speak the formed word
- **pyttsx3 Engine**: Cross-platform speech synthesis
- **Adjustable Voice**: Customize speed and voice type
- **Clear Pronunciation**: Natural-sounding speech

### 🔤 **Special Gestures**
- ✅ **Space**: Add space between words
- ✅ **Delete**: Remove last character
- ✅ **Nothing**: Neutral/rest position
- ✅ **A-Z Letters**: Full alphabet support

### 📊 **Machine Learning Pipeline**
- **Random Forest Classifier**: 100 estimators
- **Feature Engineering**: Normalized hand coordinates (42 features)
- **Train/Test Split**: 80/20 with stratification
- **Model Persistence**: Pickle serialization
- **High Accuracy**: 95%+ on test data

### 🎥 **Real-time Interface**
- **Live Video Feed**: Webcam integration
- **Hand Visualization**: Green bounding boxes
- **Character Display**: Real-time predictions
- **Word Display**: Accumulated text
- **Landmark Drawing**: Visual hand skeleton

---

## 🏗️ Architecture

### System Workflow

```mermaid
graph TB
    A[Dataset Images] --> B[extract.py]
    B --> C[Hand Landmarks 21x2]
    C --> D[data.pickleee]
    D --> E[train.py]
    E --> F[Random Forest Model]
    F --> G[model.p]
    G --> H[test.py / real.py]
    H --> I[Webcam Input]
    I --> J[MediaPipe Processing]
    J --> K[Hand Detection]
    K --> L[Gesture Prediction]
    L --> M[Word Building]
    M --> N[Text-to-Speech]
    style A fill:#e3f2fd
    style N fill:#c8e6c9
```

### Processing Pipeline

```
┌──────────────────────────────────────────────┐
│         Data Extraction (extract.py)         │
├──────────────────────────────────────────────┤
│  1. Load ASL Images                          │
│  2. MediaPipe Hand Detection                 │
│  3. Extract 21 Landmarks (x, y)              │
│  4. Normalize Coordinates                    │
│  5. Save to data.pickleee (42 features)      │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│         Model Training (train.py)            │
├──────────────────────────────────────────────┤
│  1. Load Processed Data                      │
│  2. Train/Test Split (80/20)                 │
│  3. Random Forest Classifier                 │
│  4. Model Evaluation                         │
│  5. Save to model.p                          │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│      Real-time Detection (test.py)           │
├──────────────────────────────────────────────┤
│  1. Webcam Capture                           │
│  2. MediaPipe Hand Tracking                  │
│  3. Extract Live Landmarks                   │
│  4. Model Prediction                         │
│  5. Confirmation System (15 frames)          │
│  6. Word Building Logic                      │
│  7. Text-to-Speech (Press 'S')               │
└──────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Computer Vision** | OpenCV 4.x | Video capture & processing |
| **Hand Tracking** | MediaPipe | 21-point hand landmark detection |
| **ML Algorithm** | Random Forest (sklearn) | Gesture classification |
| **TTS Engine** | pyttsx3 | Text-to-speech conversion |
| **Data Processing** | NumPy | Numerical operations |
| **Serialization** | Pickle | Model & data persistence |
| **Language** | Python 3.8+ | Core development |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- ASL Dataset ([Download here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))

### Step 1: Clone the Repository

```bash
git clone https://github.com/janaelpardisi/asl-sign-language-detection.git
cd asl-sign-language-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install opencv-python mediapipe scikit-learn numpy pyttsx3
```

**Or create `requirements.txt`:**

```txt
opencv-python>=4.8.0
mediapipe>=0.10.0
scikit-learn>=1.3.0
numpy>=1.24.0
pyttsx3>=2.90
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

1. Download ASL Alphabet Dataset from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Extract to project folder
3. Update path in `extract.py`:

```python
DATA_DIR = r"path/to/asl_alphabet_train/asl_alphabet_train"
```

---

## 💻 Usage

### Complete Pipeline (Step-by-Step)

#### **Step 1: Extract Features from Dataset**

```bash
python extract.py
```

**What it does:**
- Loads ASL images (200 per class)
- Detects hand landmarks using MediaPipe
- Extracts 21 (x, y) coordinates = 42 features
- Normalizes coordinates relative to hand bounding box
- Saves to `data.pickleee`

**Output:**
```
model saved in data.pickleee
```

---

#### **Step 2: Train the Model**

```bash
python train.py
```

**What it does:**
- Loads processed data
- Splits into train/test (80/20)
- Trains Random Forest classifier (100 trees)
- Evaluates accuracy
- Saves model to `model.p`

**Output:**
```
Data shape: (5200, 42)
Labels shape: (5200,)
Accuracy: 97.31%
Model saved as 'model.p'
```

---

#### **Step 3: Run Real-time Detection**

**Option A: Simple Detection (real.py)**

```bash
python real.py
```

- Basic gesture recognition
- Shows predicted character
- No word building

**Option B: Advanced Detection (test.py)**

```bash
python test.py
```

- Full word building system
- Character confirmation
- Text-to-speech
- Special commands (space, delete)

**Controls:**
- **'S' Key**: Speak the word (text-to-speech)
- **'Q' Key**: Quit application

---

### How to Use the System

1. **Position Your Hand**
   - Place hand in front of webcam
   - Ensure good lighting
   - Keep hand within frame

2. **Make a Gesture**
   - Form ASL letter with your hand
   - Hold gesture steady for ~0.5 seconds
   - Green box appears around detected hand

3. **Build Words**
   - Characters automatically added to word
   - Use "space" gesture for spaces
   - Use "del" gesture to delete last character

4. **Speak the Word**
   - Press 'S' key
   - System speaks the formed word
   - Word resets after speaking

---

## 📁 Project Structure

```
asl-sign-language-detection/
│
├── extract.py              # Extract features from dataset
├── train.py                # Train Random Forest model
├── test.py                 # Advanced detection (word building + TTS)
├── real.py                 # Simple real-time detection
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore file
│
├── data.pickleee          # Processed features (generated)
├── model.p                # Trained model (generated)
│
└── asl_alphabet_train/    # ASL dataset (download separately)
    ├── A/
    ├── B/
    ├── C/
    └── ...
```

---

## 🔧 Configuration

### Adjusting Dataset Size

In `extract.py`:

```python
# Load fewer/more images per class
img_paths = os.listdir(os.path.join(DATA_DIR, dir_))[:200]  # Change 200
```

### Tuning Model Parameters

In `train.py`:

```python
model = RandomForestClassifier(
    n_estimators=100,      # Number of trees (50-200)
    max_depth=None,        # Tree depth (None = unlimited)
    min_samples_split=2,   # Min samples to split
    random_state=42
)
```

### Adjusting Confirmation Threshold

In `test.py`:

```python
if confirm_counter > 15:  # Change threshold (10-30 frames)
    # Add character
```

### Changing Detection Confidence

In `extract.py` and `test.py`:

```python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,           # Detect 1 or 2 hands
    min_detection_confidence=0.3  # Lower = more detections (0.1-0.9)
)
```

### Customizing Text-to-Speech

In `test.py`:

```python
engine = pyttsx3.init()

# Adjust speech rate
engine.setProperty('rate', 150)  # Default: 200

# Change voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice

# Adjust volume
engine.setProperty('volume', 0.9)  # 0.0 to 1.0
```

---

## 🔄 How It Works

### Feature Extraction Process

1. **Image Loading**: Read ASL gesture image
2. **Hand Detection**: MediaPipe identifies hand in image
3. **Landmark Extraction**: 21 keypoints detected (fingertips, joints, palm)
4. **Coordinate Collection**: (x, y) for each landmark
5. **Normalization**: 
   ```python
   normalized_x = x - min(x_)  # Relative to leftmost point
   normalized_y = y - min(y_)  # Relative to topmost point
   ```
6. **Feature Vector**: 42 values (21 points × 2 coordinates)

### Classification Process

1. **Input**: 42 normalized hand coordinates
2. **Random Forest**: 100 decision trees vote
3. **Prediction**: Most voted class wins
4. **Output**: Predicted character (A-Z, space, del, nothing)

### Word Building Logic

```python
# Confirmation system
if predicted_character == last_character:
    confirm_counter += 1
else:
    confirm_counter = 0

# Add character after 15 frames
if confirm_counter > 15:
    if predicted_character == 'space':
        word += ' '
    elif predicted_character == 'del':
        word = word[:-1]  # Remove last char
    elif predicted_character not in ['nothing']:
        word += predicted_character
    confirm_counter = 0
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'mediapipe'`

**Solution**:
```bash
pip install mediapipe
```

---

**Issue**: `FileNotFoundError: data.pickleee`

**Solution**:
```bash
# Run extract.py first to generate data
python extract.py
```

---

**Issue**: `Webcam not detected`

**Solution**:
```python
# Try different camera index
cap = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc.
```

---

**Issue**: Low accuracy / wrong predictions

**Solutions**:
1. **Improve lighting**: Use bright, even lighting
2. **Clean background**: Reduce visual clutter
3. **Train with more data**: Increase dataset size
4. **Adjust confidence**: Lower `min_detection_confidence`
5. **Retrain model**: Run `train.py` again

---

**Issue**: Hand landmarks not detected

**Solution**:
```python
# Lower confidence threshold
hands = mp_hands.Hands(min_detection_confidence=0.1)

# Ensure good contrast between hand and background
```

---

**Issue**: Text-to-speech not working

**Solution**:
```bash
# Install/reinstall pyttsx3
pip uninstall pyttsx3
pip install pyttsx3

# Check audio output device is enabled
```

---

**Issue**: `ValueError: X has 42 features but model expects different`

**Solution**:
```python
# Ensure data preprocessing is consistent
# Check that all samples have exactly 42 features (21 landmarks × 2)
if len(data_aux) == 42:
    data.append(data_aux)
```

---

## 📊 Performance

### Model Metrics

- **Accuracy**: 95-98% on test set
- **Training Time**: ~5-10 seconds (5000 samples)
- **Inference Time**: <10ms per frame
- **FPS**: 25-30 on standard webcam

### Hardware Requirements

**Minimum:**
- CPU: Intel i3 or equivalent
- RAM: 4GB
- Webcam: 720p

**Recommended:**
- CPU: Intel i5 or better
- RAM: 8GB
- Webcam: 1080p
- Good lighting conditions

---

## 🎨 Customization Examples

### Adding New Gestures

1. **Collect images** for new gesture
2. **Add folder** to dataset: `asl_alphabet_train/NEW_GESTURE/`
3. **Run extract.py** to process
4. **Retrain model**: `python train.py`

### Changing UI Colors

In `test.py`:

```python
# Green box around hand
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
# Change to blue: (255, 0, 0)

# Red predicted character
cv2.putText(frame, text, position, font, 1.5, (0, 0, 255), 3)
# Change to yellow: (0, 255, 255)
```

### Export Word to File

```python
# Add this after forming word
if cv2.waitKey(1) & 0xFF == ord('e'):  # Press 'E' to export
    with open('output.txt', 'a') as f:
        f.write(word + '\n')
    print(f"Saved: {word}")
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/NewGesture`)
3. **Commit** your changes (`git commit -m 'Add NewGesture'`)
4. **Push** to the branch (`git push origin feature/NewGesture`)
5. **Open** a Pull Request

### Ideas for Contribution

- [ ] Add more sign languages (BSL, ISL, etc.)
- [ ] Implement sentence prediction
- [ ] Add gesture recording mode
- [ ] Create GUI with Tkinter/PyQt
- [ ] Implement two-hand gestures
- [ ] Add gesture history tracking
- [ ] Create mobile app version
- [ ] Add auto-correction feature
- [ ] Implement gesture smoothing
- [ ] Add dark mode UI

---

## 📝 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Jana Ashraf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 👨‍💻 Author

**Jana Ashraf**
- GitHub: [@janaelpardisi](https://github.com/janaelpardisi)
- LinkedIn: [Jana Ashraf](https://www.linkedin.com/in/jana-ashraf-elpardisi)

---

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) - For hand tracking solution
- [OpenCV](https://opencv.org/) - For computer vision tools
- [Scikit-learn](https://scikit-learn.org/) - For ML algorithms
- [ASL Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) - For training data
- Deaf and hard-of-hearing community for inspiration

---

## 📈 Roadmap

**Current Version**: v1.0

**Upcoming Features**:
- [ ] Support for ASL phrases/words
- [ ] Two-hand gesture recognition
- [ ] Real-time translation to multiple languages
- [ ] Mobile app (iOS/Android)
- [ ] Web-based interface
- [ ] Cloud deployment
- [ ] Sign language learning mode
- [ ] Video recording & playback
- [ ] Social sharing features
- [ ] Integration with video conferencing tools

---

## 💡 Use Cases

### Education
- Sign language learning tool
- Interactive ASL tutorials
- Classroom accessibility
- Student projects

### Accessibility
- Communication aid for deaf/HoH
- Public service kiosks
- Healthcare settings
- Emergency services

### Research
- Gesture recognition studies
- Human-computer interaction
- Machine learning education
- Computer vision demos

### Entertainment
- Sign language games
- Interactive exhibits
- Virtual reality integration
- Educational apps

---

## 🔒 Privacy & Ethics

- **No Data Storage**: Hand tracking data not saved
- **Local Processing**: All computation happens on device
- **No Cloud Upload**: Video never leaves your computer
- **Open Source**: Fully auditable code
- **Respectful**: Built to empower, not replace, ASL users

---

<div align="center">

**Made with Jana❤️**

**🤟 Breaking Communication Barriers with AI**

[⬆ Back to Top](#-asl-sign-language-detection-system)

</div>
