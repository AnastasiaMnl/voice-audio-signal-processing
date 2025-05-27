import librosa
import numpy as np
import os

def extract_features(file_path, sr=16000, frame_len=0.025, hop_len=0.01, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    frame_length = int(sr * frame_len)
    hop_length = int(sr * hop_len)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=frame_length, hop_length=hop_length)
    return mfcc.T  # (frames, features)

def create_dataset(speech_dir, noise_dir):
    X = []
    y = []

    for root, _, files in os.walk(speech_dir):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                features = extract_features(path)
                X.append(features)
                y.append(np.ones(features.shape[0]))  # foreground

    for root, _, files in os.walk(noise_dir):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                features = extract_features(path)
                X.append(features)
                y.append(np.zeros(features.shape[0]))  # background

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y
