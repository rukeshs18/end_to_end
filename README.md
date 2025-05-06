# 🎙️ End-to-End Speech Recognition Using Deep Learning

This project implements a basic end-to-end speech recognition system using the LibriSpeech dataset. The goal is to recognize speech by converting audio signals into text using deep learning techniques. The pipeline includes data loading, preprocessing with MFCC feature extraction, label encoding, and training a hybrid DNN-LSTM model using TensorFlow/Keras.

## 📁 Dataset

The project uses a subset of the **LibriSpeech `dev-clean` dataset**, which contains `.flac` audio files and corresponding text transcriptions. You can download it from: [LibriSpeech ASR Corpus](http://www.openslr.org/12)

## 🔧 Features

- Loads audio files and corresponding transcripts from LibriSpeech.
- Preprocesses audio by resampling and extracting MFCC features.
- Encodes transcripts using `LabelEncoder`.
- Builds a Sequential Deep Learning Model:
  - Dense layers + Dropout
  - Bidirectional LSTM + LSTM layers
- Trains and evaluates the model using accuracy and loss metrics.
- Visualizes model accuracy over epochs.

## 🧠 Model Architecture

```
Input: MFCC features (13 coefficients)
|
-> Dense (128) + ReLU
-> Dropout (0.2)
-> Dense (64) + ReLU
-> Bidirectional LSTM (128, return_sequences=True)
-> Dropout (0.2)
-> LSTM (128)
-> Dense (Softmax, number of unique transcripts)
```

## 📊 Training Output

- Model is trained for 10 epochs using an 80/20 train-test split.
- Training and validation accuracy are plotted for visual analysis.
- Final test accuracy and loss are reported.

## 🚀 Getting Started

### 📦 Requirements

- Python 3.7+
- TensorFlow
- NumPy
- Librosa
- SoundFile
- scikit-learn
- Matplotlib

```bash
pip install tensorflow librosa soundfile scikit-learn matplotlib
```

### ▶️ Run the Project

1. Update `DATA_DIR` in the script to point to your local LibriSpeech `dev-clean` path.
2. Run the Python script:

```bash
python end.py
```

## 📈 Results

- The model demonstrates basic speech recognition capability.
- Designed for educational/demo purposes due to limited dataset size (20 samples).

## 🔮 Future Improvements

- Use CTC (Connectionist Temporal Classification) loss for better sequence alignment.
- Scale dataset for better generalization.
- Implement character/word-level tokenization instead of full transcript classification.
- Integrate with a real-time speech input system.

## 📄 License

This project is open-source and available under the MIT License.
