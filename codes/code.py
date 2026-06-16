# =====================================================
# AUDIO ANALYSIS USING DEEP LEARNING
# =====================================================

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# =====================================================
# LOAD AUDIO DATASET
# =====================================================

dataset_path = "dataset"

features = []
labels = []

print("Loading Audio Files...")

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                file_path = os.path.join(folder_path, file)

                try:

                    audio, sample_rate = librosa.load(
                        file_path,
                        res_type='kaiser_fast'
                    )

                    mfccs = librosa.feature.mfcc(
                        y=audio,
                        sr=sample_rate,
                        n_mfcc=40
                    )

                    mfccs_scaled = np.mean(
                        mfccs.T,
                        axis=0
                    )

                    features.append(mfccs_scaled)
                    labels.append(folder)

                except Exception as e:
                    print("Error:", file_path)

print("Audio Loading Complete")

# =====================================================
# CREATE DATAFRAME
# =====================================================

X = np.array(features)
y = np.array(labels)

print("Feature Shape:", X.shape)

# =====================================================
# LABEL ENCODING
# =====================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# =====================================================
# DEEP LEARNING MODEL
# =====================================================

model = Sequential()

model.add(
    Dense(
        256,
        activation='relu',
        input_shape=(40,)
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        64,
        activation='relu'
    )
)

model.add(
    Dense(
        y_categorical.shape[1],
        activation='softmax'
    )
)

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# =====================================================
# EVALUATION
# =====================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy)

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# =====================================================
# LOSS GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# =====================================================
# SAVE MODEL
# =====================================================

model.save(
    "audio_classifier.h5"
)

print("Model Saved Successfully")

# =====================================================
# LOAD MODEL
# =====================================================

saved_model = load_model(
    "audio_classifier.h5"
)

# =====================================================
# AUDIO PREDICTION
# =====================================================

def predict_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        res_type='kaiser_fast'
    )

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfccs_scaled = np.mean(
        mfccs.T,
        axis=0
    )

    mfccs_scaled = mfccs_scaled.reshape(
        1,
        -1
    )

    prediction = saved_model.predict(
        mfccs_scaled
    )

    predicted_class = np.argmax(
        prediction
    )

    return encoder.inverse_transform(
        [predicted_class]
    )[0]

# =====================================================
# TEST PREDICTION
# =====================================================

test_audio = "sample.wav"

if os.path.exists(test_audio):

    result = predict_audio(test_audio)

    print("\nPredicted Audio Class:")
    print(result)

else:

    print(
        "\nPlace a sample.wav file in project folder "
        "to test prediction."
    )

# =====================================================
# END
# =====================================================

print("\nAudio Analysis Completed Successfully")