# Project for audio processing course
# COMP.SGN.120-2023-2024-1 Introduction to Audio Processing

# Name: Arttu Raatikainen
# Student number: H291629
# email: arttu.raatikainen@tuni.fi

"""
This is a demo version of the real project. The real project is done with notebook.
Demo version contains only the plotting functions and the audio files are not included.
Neither are audio file normalizing, Model training and Model predictions.

"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


# Add path to audio files for bus and car
busFile = ""
carFile = ""

# define plotting functions

def plot_mel_spectrogram(x, fs, n_mels=128, name="Mel Spectrogram"):

    melSpec = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_mels)
    melSpecDb = librosa.amplitude_to_db(abs(melSpec), ref=np.max)

    plt.figure( figsize=(10, 5) )
    librosa.display.specshow(melSpecDb, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB', label='Amplitude')
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel("Mels")
    plt.show()

def plot_mfcc(x, fs, n_mfcc, name="MFCC"):

    mfccs = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mfcc)

    plt.figure( figsize=(10, 5) )
    librosa.display.specshow(mfccs, x_axis='time', norm=Normalize(vmin=-20, vmax=20))
    plt.colorbar(format='%+2.0f dB')
    plt.title(name)
    plt.yticks()
    plt.xlabel("Time ")
    plt.ylabel("MFC Coefficients")
    plt.show()

def plot_zero_crossing_rate(x, fs, frame_length, hop_length, name="Zero Crossing Rate"):
    # calculate zero crossing rate for length of audio signal
    zcr = librosa.feature.zero_crossing_rate(y=x, frame_length=frame_length, hop_length=hop_length)[0]
    maxVal = np.max(np.abs(zcr))
    zcr = np.array(zcr) / maxVal if maxVal > 0 else zcr
    time_zcr = np.linspace(0, len(x)/fs, len(zcr))

    plt.figure( figsize=(10, 5) )
    plt.plot(time_zcr, zcr, color='r', label='Zero Crossing Rate')
    plt.title(name)
    plt.xlabel("Frames")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()

def plot_spectral_centroid(x, fs, name="Spectral Centroid"):

    X = librosa.stft(x.astype(float))
    X_mag = np.abs(X)
    spectral_centroids = librosa.feature.spectral_centroid(S=X_mag, sr=fs)[0]
    spectral_centroids = np.array(spectral_centroids) / np.max(spectral_centroids) if np.max(spectral_centroids) > 0 else spectral_centroids
    t = librosa.frames_to_time(range(len(spectral_centroids)))

    plt.figure( figsize=(10, 5) )
    plt.plot(t, spectral_centroids, color='y')
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(('Audio Signal', 'RMS', 'ZCR', 'Spectral Centroid'))
    plt.show()

def plot_rmse(x, fs, frame_length, hop_length, name="RMSE"):

    # calculate RMS Energy RMSE for length of audio signal
    rmse = []

    for i in range(0, len(x), hop_length):
        segment = x[i:i+frame_length]
        rmseVal = np.sqrt(np.mean(segment**2))
        rmse.append(rmseVal)
    
    #plot RMS in red with audio signal
    time_rmse = np.arange(len(rmse)) * hop_length / fs


    plt.figure( figsize=(10, 5) )
    plt.plot(time_rmse, rmse, color='g', label='RMSE')
    plt.title(name)
    plt.xlabel("Frames")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def plot_signal(x, fs, name="Audio Signal"):

    plt.figure( figsize=(10, 5) )
    plt.plot(x)
    plt.title(name)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_chromagram(x, fs, name="Chromagram"):
    # calculate chromagram
    X = librosa.stft(x.astype(float))
    X_db = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    chromagram = librosa.feature.chroma_stft(S=X_db, sr=fs)
    
    plt.figure( figsize=(10, 5) )
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar(label='Relative Intensity')
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel("Pitch Class")
    plt.show()

# read and plot audio files

carX, carFs = librosa.load(carFile)
busX, busFs = librosa.load(busFile)

plot_signal(carX, carFs, "Car Audio Signal")
plot_signal(busX, busFs, "Bus Audio Signal")

plot_mel_spectrogram(carX, carFs, name="Car Mel Spectrogram")
plot_mel_spectrogram(busX, busFs, name="Bus Mel Spectrogram")

plot_mfcc(carX, carFs, 15, name="Car MFCC")
plot_mfcc(busX, busFs, 15, name="Bus MFCC")

plot_zero_crossing_rate(carX, carFs, 1024, 512, name="Car Zero Crossing Rate")
plot_zero_crossing_rate(busX, busFs, 1024, 512, name="Bus Zero Crossing Rate")

plot_spectral_centroid(carX, carFs, name="Car Spectral Centroid")
plot_spectral_centroid(busX, busFs, name="Bus Spectral Centroid")

plot_rmse(carX, carFs, 1024, 512, name="Car RMSE")
plot_rmse(busX, busFs, 1024, 512, name="Bus RMSE")

plot_chromagram(carX, carFs, name="Car Chromagram")
plot_chromagram(busX, busFs, name="Bus Chromagram")



