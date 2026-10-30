import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# 1. Load audio
signal, fs = sf.read("voice.wav")
if signal.ndim > 1:
    signal = signal.mean(axis=1)

# Resample or verify rate (LPC works best around 10-16 kHz)
# Target speech segment around 3.0s (sustained voiced vowel)
center_sample = int(3.0 * fs)
frame_len = int(0.03 * fs)  # 30 ms
frame = signal[center_sample : center_sample + frame_len]

# 2. Pre-emphasis filter to boost higher frequencies (+6 dB/octave)
pre_emphasis = 0.97
emphasized_frame = np.append(frame[0], frame[1:] - pre_emphasis * frame[:-1])
windowed = emphasized_frame * np.hamming(len(frame))

# 3. LPC Analysis
# Rule of thumb for LPC order: fs / 1000 + 2 -> 16 + 2 = 18 for 16 kHz
lpc_order = int(fs / 1000) + 2
# Autocorrelation method for LPC coefficients (Levinson-Durbin)
r = np.correlate(windowed, windowed, mode='full')[len(windowed)-1:]
a = np.zeros(lpc_order + 1)
# Solve Yule-Walker via Levinson-Durbin
from scipy.linalg import solve_toeplitz
a[0] = 1.0
a[1:] = solve_toeplitz((r[:lpc_order], r[:lpc_order]), -r[1:lpc_order+1])

# 4. Find roots of LPC polynomial to locate resonant poles
roots = np.roots(a)
# Keep roots with positive imaginary parts (upper half of z-plane)
roots = [r for r in roots if np.imag(r) >= 0.01 and np.abs(r) > 0.7]

# 5. Convert root angles to physical frequencies in Hz
formants = []
for r in roots:
    freq = np.arctan2(np.imag(r), np.real(r)) * (fs / (2.0 * np.pi))
    if 200 < freq < 4000:
        formants.append(freq)
formants = sorted(formants)

print(f"Detected Formants at t=3.0s:")
for i, f in enumerate(formants[:3], 1):
    print(f"  F{i}: {f:.1f} Hz")

# 6. Plot LPC Spectral Envelope against the FFT Spectrum
freqs, h = np.fft.rfftfreq(1024, 1/fs), np.fft.rfft(windowed, 1024)
fft_db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))

w, h_lpc = np.fft.rfftfreq(1024, 1/fs), 1.0 / np.fft.rfft(a, 1024)
lpc_db = 20 * np.log10(np.maximum(np.abs(h_lpc), 1e-5))
# Offset LPC curve to overlay cleanly over the FFT
lpc_db += (np.max(fft_db) - np.max(lpc_db))

plt.figure(figsize=(9, 4))
plt.plot(freqs, fft_db, color="gray", alpha=0.5, label="FFT Harmonics")
plt.plot(w, lpc_db, color="navy", linewidth=2, label="LPC Vocal Tract Envelope")
for f in formants[:3]:
    plt.axvline(f, color="crimson", linestyle="--", alpha=0.8)
plt.xlim(0, 4000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("LPC Spectral Envelope & Formant Resonances")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()