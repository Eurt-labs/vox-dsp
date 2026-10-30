#Track Fundamental Pitch (f0) Over Time
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# 1. Load the digitized audio file
signal, fs = sf.read("voice.wav")
if signal.ndim > 1:
    signal = signal.mean(axis=1)  # Convert stereo to mono

# 2. Pitch extraction function
def extract_pitch_contour(signal, fs, frame_size=512, hop_size=256, fmin=75, fmax=400):
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)
    
    num_frames = (len(signal) - frame_size) // hop_size
    pitches = []
    time_stamps = []
    
    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start : start + frame_size] * np.hanning(frame_size)
        
        # Energy threshold to skip unvoiced/silent frames
        energy = np.sum(frame ** 2)
        if energy < 1e-3:
            pitches.append(np.nan)
        else:
            # Normalized autocorrelation
            corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
            search_window = corr[min_lag:max_lag]
            peak_lag = min_lag + np.argmax(search_window)
            pitches.append(fs / peak_lag)
            
        time_stamps.append((start + frame_size / 2) / fs)
        
    return np.array(time_stamps), np.array(pitches)

# 3. Extract and plot pitch contour
times, f0 = extract_pitch_contour(signal, fs)

plt.figure(figsize=(10, 3))
plt.plot(times, f0, color="crimson", linewidth=1.5, label="Fundamental Pitch (f0)")
plt.xlabel("Time (s)")
plt.ylabel("Pitch (Hz)")
plt.ylim(50, 450)
plt.title("Vocal Pitch Tracking Contour")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()