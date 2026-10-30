import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# 1. Load audio
signal, fs = sf.read("voice.wav")
if signal.ndim > 1:
    signal = signal.mean(axis=1)

# 2. Slice into 32 ms frames and locate the loudest frame (highest RMS energy)
frame_len = 512
hop_len = 256
num_frames = (len(signal) - frame_len) // hop_len

frame_energies = [
    np.sum(signal[i * hop_len : i * hop_len + frame_len] ** 2)
    for i in range(num_frames)
]
loudest_frame_idx = np.argmax(frame_energies) * hop_len
frame = signal[loudest_frame_idx : loudest_frame_idx + frame_len]

# 3. Apply Hann window and run DFT
window = np.hanning(frame_len)
dft = np.fft.rfft(frame * window)
mag_db = 20 * np.log10(np.maximum(np.abs(dft) / frame_len, 1e-10))
freqs = np.fft.rfftfreq(frame_len, d=1.0 / fs)

# 4. Find the dominant voiced frequency
peak_idx = np.argmax(mag_db)
print(f"Loudest Frame at: {loudest_frame_idx / fs:.2f} s")
print(f"Strongest Peak: {freqs[peak_idx]:.1f} Hz at {mag_db[peak_idx]:.1f} dB")

# 5. Plot the spectrum up to 4000 Hz (speech band)
plt.figure(figsize=(9, 4))
plt.plot(freqs, mag_db, color="#1f77b4", linewidth=1.2)
plt.axvline(freqs[peak_idx], color="red", linestyle="--", label=f"Peak: {freqs[peak_idx]:.1f} Hz")
plt.xlim(0, 4000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Active Speech DFT Spectrum")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()