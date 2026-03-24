# Real-Time Emotion Recognition System

## 📖 Introduction
Emotion recognition is a sub-field of affective computing that aims to identify human emotions from facial expressions. This project implements a real-time emotion recognition system using deep learning techniques to detect and classify emotions continuously through a webcam feed. The model is capable of understanding facial nuances to predict the user's emotional state in real-time.

## 🎯 Problem Statement
Human emotion recognition is vital for improving human-computer interaction, mental health monitoring, and personalized user experiences. The challenge lies in developing a system that accurately and efficiently interprets complex facial expressions under varying lighting conditions, orientations, and scales in a live environment.

## 📌 Objectives
- To design and implement a Convolutional Neural Network (CNN) architecture capable of feature extraction from facial images.
- To train a robust deep learning model on a vast dataset of varied human facial expressions.
- To integrate the trained model with OpenCV for real-time video stream processing and emotion classification.
- To achieve a competitive accuracy rate suitable for real-world academic and practical applications.

## 🛠 Features & Technologies Used
- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Computer Vision Library:** OpenCV
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

## 📊 Dataset Description
The model is trained on the widely recognized **FER-2013 (Facial Expression Recognition 2013)** dataset.
- **Content:** The dataset consists of 48x48 pixel grayscale images of faces.
- **Classes:** The faces have been automatically registered so that the face is roughly centered and occupies about the same amount of space in each image.
- **Emotion Categories (7 Classes):** 
  - Angry 😡
  - Disgust 🤢
  - Fear 😨
  - Happy 😄
  - Sad 😢
  - Surprise 😲
  - Neutral 😐

## ⚙️ System Design & Pipeline
1. **Data Acquisition:** Load the FER-2013 dataset images and corresponding emotion labels.
2. **Data Preprocessing:** Convert images to grayscale, resize to 48x48 pixels, normalize pixel values (0-1), and perform data augmentation to prevent overfitting.
3. **Model Construction:** Build a CNN sequential model with multiple Convolutional layers, MaxPooling layers, Dropout layers for regularization, and Fully Connected Dense layers.
4. **Model Training:** Train the CNN model utilizing categorical cross-entropy loss and an adam optimizer over multiple epochs.
5. **Real-Time Integration:** Utilize OpenCV's Haar Cascade classifier to detect faces in a live camera feed.
6. **Prediction:** Extract the localized face ROI, preprocess it, and feed it to the trained CNN model to output the recognized emotion class text onto the video frame.

## 🚀 Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/riishaal/EmotionDetection_60.git
   cd EmotionDetection_60
   ```

2. **Create a virtual environment (Optional but Recommended):**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install tensorflow keras opencv-python numpy pandas matplotlib
   ```
   *(Alternatively, if you have a `requirements.txt` file setup: `pip install -r requirements.txt`)*

## 🏃‍♂️ Execution Steps

The project execution is broken down into three main scripts. Run them in the following sequence:

1. **Preprocess the Data:**
   Use this script to analyze, clean, and augment the dataset before training.
   ```bash
   python preprocess.py
   ```

2. **Train the Model:**
   Execute the training script to build and train the CNN model. This script will save the trained weights locally (e.g., `model.h5`).
   ```bash
   python train_model.py
   ```

3. **Run Real-Time Detection:**
   Launch the webcam interface to predict emotions in real-time.
   ```bash
   python realtime_detection.py
   ```
   *Note: Press **'q'** in the OpenCV video window to safely quit the prediction stream.*

## 📈 Output Explanation & Results
- **Output:** The system launches a live video feed, drawing a bounding box around detected faces and displaying the predicted emotion label directly above the bounding box.
- **Results:** The CNN model achieves an approximate validation accuracy of **55%**, which is highly competitive and standard given the highly subjective, noisy, and challenging nature of the FER-2013 dataset.

## 🚧 Challenges Faced
- **Class Imbalance:** The dataset contained imbalanced and heavily skewed class distributions (e.g., far fewer 'Disgust' images), requiring specific augmentation and class-weighting adjustments.
- **Hardware Limitations:** Training deep CNN architectures naturally requires significant computational power; overcoming out-of-memory errors involved optimizing image batch sizes.
- **Varying Practical Conditions:** Real-time webcam feeds frequently introduced varying lighting setups and head poses that slightly impacted real-world inference stability compared to static dataset evaluations.

## 🔮 Future Scope
- **Architectural Enhancements:** Implementing Deep Transfer Learning architectures like ResNet, VGG16, or MobileNet to boost the baseline prediction accuracy.
- **Multi-Face recognition:** Enhancing the OpenCV pipeline to accurately track and distinctively label multiple faces dynamically within crowded frames.
- **Web Application:** Deploying the trained model onto a cloud server and building a user-friendly web interface using frameworks like Flask or Streamlit.
- **Multi-Modal Detection:** Combining audio streams (voice tone and pitch) with visual cues for a highly robust, compound emotion recognition engine.

## 📚 References
- Goodfellow, I., et al. (2013). Challenges in Representation Learning: A report on three machine learning contests. [Link](https://arxiv.org/abs/1307.0414)
- OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
- Keras Documentation: [https://keras.io/](https://keras.io/)
- FER-2013 Dataset on Kaggle: [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

---
*Developed by [MUHAMMED RISHAL K](https://github.com/riishaal)*
