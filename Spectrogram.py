import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# 1. Load audio
signal, fs = sf.read("voice.wav")
if signal.ndim > 1:
    signal = signal.mean(axis=1)

# 2. Compute STFT Spectrogram
# nperseg = 512 samples (32 ms window)
# noverlap = 384 samples (75% overlap for smooth time resolution)
frequencies, times, Sxx = spectrogram(
    signal, 
    fs=fs, 
    window="hann", 
    nperseg=512, 
    noverlap=384, 
    scaling="spectrum"
)

# 3. Convert magnitude to decibels
spectrogram_db = 10 * np.log10(np.maximum(Sxx, 1e-10))

# 4. Plot Spectrogram limited to speech band (0 to 4000 Hz)
plt.figure(figsize=(10, 5))
plt.pcolormesh(times, frequencies, spectrogram_db, shading="gouraud", cmap="magma")
plt.ylim(0, 4000)
plt.colorbar(label="Magnitude (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Speech Spectrogram (STFT)")
plt.tight_layout()
plt.show()