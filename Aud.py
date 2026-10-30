import numpy as np
import soundfile as sf

# 1. Load digitized audio (ADC output normalized to [-1.0, 1.0])
# Replace 'voice.wav' with your recording, or synthesize a test vowel frame below
signal, fs = sf.read("voice.wav")
if signal.ndim > 1:
    signal = signal.mean(axis=1)  # Convert stereo to mono

# 2. Slice a single 32 ms stationary frame (N = 512 samples at fs = 16000 Hz)
N = 512
start_idx = len(signal) // 2      # Grab a frame from the middle of the clip
frame = signal[start_idx : start_idx + N]

# 3. Apply a Hann window to suppress spectral leakage at frame edges
window = np.hanning(N)
windowed_frame = frame * window

# 4. Compute the Real DFT (rFFT returns N/2 + 1 positive frequency bins)
dft_complex = np.fft.rfft(windowed_frame)
magnitude = np.abs(dft_complex) / N
magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))

# 5. Map bin indices k = [0, 1, ..., N/2] to physical frequencies in Hz
freqs_hz = np.fft.rfftfreq(N, d=1.0 / fs)

# 6. Find the dominant frequency peak (often the fundamental pitch or 1st formant)
peak_bin = np.argmax(magnitude_db)
print(f"Sampling Rate: {fs} Hz | Frame Size: {N} samples ({N/fs*1000:.1f} ms)")
print(f"Frequency Resolution (Bin Width): {fs/N:.2f} Hz")
print(f"Strongest Spectral Peak: {freqs_hz[peak_bin]:.1f} Hz at {magnitude_db[peak_bin]:.1f} dB")