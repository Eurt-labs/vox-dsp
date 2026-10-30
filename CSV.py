import csv
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.linalg import solve_toeplitz
from analyze_voice import analyze_audio_signal

audio_file = "voice.wav"
csv_file = "voice_dataset.csv"

# 1. Extract the 25-feature profile and append to CSV
profile = analyze_audio_signal(audio_file)
profile["File"] = audio_file

file_exists = os.path.isfile(csv_file)
with open(csv_file, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(profile.keys()))
    if not file_exists:
        writer.writeheader()
    writer.writerow(profile)

# 2. Load audio for unified multi-panel visualization
signal, fs = librosa.load(audio_file, sr=None, mono=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Panel A: Waveform & RMS Energy Envelope
times_wav = np.linspace(0, len(signal) / fs, len(signal))
axes[0, 0].plot(times_wav, signal, color="steelblue", alpha=0.7, linewidth=0.7)
axes[0, 0].set_title("Time-Domain Waveform")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True, alpha=0.3)

# Panel B: STFT Spectrogram (0 - 4000 Hz)
S = librosa.stft(signal, n_fft=512, hop_length=128, window="hann")
S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
img_spec = librosa.display.specshow(S_db, sr=fs, hop_length=128, x_axis="time", y_axis="linear", ax=axes[0, 1], cmap="magma")
axes[0, 1].set_ylim(0, 4000)
axes[0, 1].set_title("STFT Spectrogram & Formant Bands")
fig.colorbar(img_spec, ax=axes[0, 1], format="%+2.0f dB")

# Panel C: LPC Spectral Envelope on Loudest Frame
frame_len = int(0.03 * fs)
frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=int(0.01 * fs))
vowel_frame = frames[:, np.argmax(np.sum(frames**2, axis=0))] * np.hamming(frame_len)
lpc_order = int(fs / 1000) + 2
r = np.correlate(vowel_frame, vowel_frame, mode="full")[frame_len - 1 :]
a = np.concatenate(([1.0], solve_toeplitz((r[:lpc_order], r[:lpc_order]), -r[1 : lpc_order + 1])))

freqs = np.fft.rfftfreq(1024, 1 / fs)
fft_db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(vowel_frame, 1024)), 1e-5))
lpc_db = 20 * np.log10(np.maximum(np.abs(1.0 / np.fft.rfft(a, 1024)), 1e-5))
lpc_db += np.max(fft_db) - np.max(lpc_db)

axes[1, 0].plot(freqs, fft_db, color="gray", alpha=0.5, label="FFT Harmonics")
axes[1, 0].plot(freqs, lpc_db, color="navy", linewidth=2, label="LPC Envelope")
axes[1, 0].set_xlim(0, 4000)
axes[1, 0].set_title("Loudest Frame LPC Vocal Tract Filter")
axes[1, 0].set_xlabel("Frequency (Hz)")
axes[1, 0].set_ylabel("Magnitude (dB)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Panel D: CMVN Normalized MFCCs (c1 - c12)
mfccs = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13, n_fft=512, hop_length=160)
mfccs_norm = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
img_mfcc = librosa.display.specshow(mfccs_norm[1:], x_axis="time", sr=fs, hop_length=160, cmap="coolwarm", ax=axes[1, 1], vmin=-2.5, vmax=2.5)
axes[1, 1].set_title("Normalized Cepstral Coefficients (c1 - c12)")
axes[1, 1].set_ylabel("MFCC Index")
fig.colorbar(img_mfcc, ax=axes[1, 1], label="Z-Score")

plt.tight_layout()
plt.savefig("voice_dashboard.png", dpi=200)
plt.show()