import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

# Path to your dataset
DATA_DIR = "C:/Users/praji/Videos/End-to-End Speech Recognition/dev-clean"

# Function to load LibriSpeech dataset
def load_librispeech_subset(data_dir, max_files=20):
    audio_text_pairs = []
    count = 0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                # Adjust transcription path based on directory structure
                txt_path = os.path.join(root, f"{file.split('-')[0]}-{file.split('-')[1]}.trans.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            parts = line.strip().split(' ', 1)
                            if parts[0] in file:
                                transcript = parts[1].lower()
                                audio, sr = sf.read(flac_path)
                                audio_text_pairs.append((audio, sr, transcript))
                                count += 1
                                break
            if count >= max_files:
                return audio_text_pairs
    return audio_text_pairs

# Example usage to load a subset of data
print("[INFO] Loading data...")
samples = load_librispeech_subset(DATA_DIR, max_files=20)
print(f"[INFO] Loaded {len(samples)} samples")

# Preprocess audio and extract MFCCs
def preprocess_audio(samples):
    features = []
    labels = []
    
    for audio, sr, transcript in samples:
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, sr, 16000)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        mfcc = np.mean(mfcc, axis=1)
        
        features.append(mfcc)
        labels.append(transcript)
    
    return np.array(features), np.array(labels)

# Process the samples
X, y = preprocess_audio(samples)

# Encode the labels (transcripts)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    LSTM(128),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()
