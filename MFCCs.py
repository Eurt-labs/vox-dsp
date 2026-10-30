import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 1. Load audio
signal, fs = librosa.load("voice.wav", sr=16000)

# 2. Extract 13 MFCCs
mfccs = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13, n_fft=512, hop_length=160)

# 3. Apply Cepstral Mean & Variance Normalization (CMVN)
# Standardizes each cepstral bin across time: mean=0, std=1
mfccs_norm = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)

# 4. Plot both: (A) Phonetic coefficients c1-c12, and (B) Frame energy c0
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Subplot 1: Phonetic Shape (c1 to c12)
img = librosa.display.specshow(
    mfccs_norm[1:], 
    x_axis="time", 
    sr=fs, 
    hop_length=160, 
    cmap="coolwarm", 
    ax=ax1,
    vmin=-2.5, 
    vmax=2.5
)
ax1.set_ylabel("MFCC Index")
ax1.set_yticks(range(0, 12))
ax1.set_yticklabels([f"c{i}" for i in range(1, 13)])
ax1.set_title("Normalized Mel-Frequency Cepstral Coefficients ($c_1 - c_{12}$)")
fig.colorbar(img, ax=ax1, label="Normalized Z-Score")

# Subplot 2: Loudness / Energy Contour (c0)
times = librosa.times_like(mfccs[0], sr=fs, hop_length=160)
ax2.plot(times, mfccs[0], color="#1f77b4", linewidth=1.2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Energy ($c_0$)")
ax2.grid(True, alpha=0.3)
ax2.set_title("Log Energy Profile ($c_0$)")

plt.tight_layout()
plt.show()