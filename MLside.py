import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define constants
SAMPLE_RATE = 22050  # Sample rate for audio files
FRAME_LENGTH = 1024   # Frame length for STFT
HOP_LENGTH = 512      # Hop length for STFT
NUM_CLASSES = 5      # Number of active classes (modify as needed)

def load_data(data_path):
    """Load .wav files and their corresponding labels."""
    audio_files = []
    labels = []  # Should contain [active class index, azimuth, elevation]
    
    for filename in os.listdir(data_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(data_path, filename)
            # Load audio file
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            audio_files.append(audio)
            # Assuming the labels are stored in a corresponding .csv file
            label_file_path = file_path.replace('.wav', '.csv')
            label_data = pd.read_csv(label_file_path)
            labels.append(label_data)
    
    return audio_files, labels

def preprocess_data(audio_files):
    """Convert audio files to Mel-spectrograms."""
    mel_spectrograms = []
    for audio in audio_files:
        mel = librosa.feature.melspectrogram(audio, sr=SAMPLE_RATE, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        mel_spectrograms.append(mel)
    return np.array(mel_spectrograms)

def build_model(input_shape):
    """Build a simple CNN model for sound event detection."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')  # Change according to the output
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test, output_path):
    """Evaluate the model and save results in CSV format."""
    y_pred = model.predict(X_test)
    # Assuming the output predictions are to be processed
    for i, pred in enumerate(y_pred):
        frame_number = i // (X_test.shape[1] // HOP_LENGTH)  # Example frame calculation
        # Saving results in CSV format
        result_df = pd.DataFrame({
            'frame_number': frame_number,
            'active_class_index': np.argmax(pred),
            'azimuth': np.random.randint(-90, 90),  # Placeholder for actual azimuth
            'elevation': np.random.randint(-90, 90)  # Placeholder for actual elevation
        }, index=[0])
        result_file = os.path.join(output_path, f'results_{i}.csv')
        result_df.to_csv(result_file, index=False)

def main(data_path, output_path):
    # Load and preprocess data
    audio_files, labels = load_data(data_path)
    X = preprocess_data(audio_files)
    y = np.array([label['active class index'] for label in labels])  # Adjust according to your labels
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model and save results
    evaluate_model(model, X_test, y_test, output_path)

# Run the main function
if __name__ == '__main__':
    data_path = 'path/to/your/wav/files'  # Specify your .wav file directory
    output_path = 'path/to/save/results'   # Specify your output directory
    main(data_path, output_path)
