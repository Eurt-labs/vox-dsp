import numpy as np
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from scipy.linalg import solve_toeplitz

def analyze_audio_signal(audio_path):
    # 1. Audio Loading & Core Specs
    signal, fs = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=signal, sr=fs)
    rms_energy = np.sqrt(np.mean(signal**2))
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

    # 2. Praat Acoustic Biometrics
    snd = parselmouth.Sound(audio_path)
    pitch_obj = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=400.0)
    mean_f0 = call(pitch_obj, "Get mean", 0, 0, "Hertz")
    std_f0 = call(pitch_obj, "Get standard deviation", 0, 0, "Hertz")

    point_proc = call(snd, "To PointProcess (periodic, cc)", 75.0, 400.0)
    jitter = call(point_proc, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100
    shimmer = call([snd, point_proc], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(snd.to_harmonicity(time_step=0.01, minimum_pitch=75.0), "Get mean", 0, 0)

    # 3. LPC Formants (on highest-energy voiced frame)
    frame_len = int(0.03 * fs)
    hop_len = int(0.01 * fs)
    frames = librosa.util.frame(signal, frame_length=frame_len, hop_length=hop_len)
    loudest_idx = np.argmax(np.sum(frames**2, axis=0))
    vowel_frame = frames[:, loudest_idx] * np.hamming(frame_len)
    
    lpc_order = int(fs / 1000) + 2
    r = np.correlate(vowel_frame, vowel_frame, mode='full')[len(vowel_frame)-1:]
    a = np.zeros(lpc_order + 1)
    a[0] = 1.0
    a[1:] = solve_toeplitz((r[:lpc_order], r[:lpc_order]), -r[1:lpc_order+1])
    
    roots = [rt for rt in np.roots(a) if np.imag(rt) >= 0.01 and np.abs(rt) > 0.7]
    formants = sorted([np.arctan2(np.imag(rt), np.real(rt)) * (fs / (2.0 * np.pi)) 
                       for rt in roots if 200 < np.arctan2(np.imag(rt), np.real(rt)) * (fs / (2.0 * np.pi)) < 4000])

    # 4. Spectral Fingerprint (13 MFCC Means)
    mfccs = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    # 5. Output Feature Dictionary
    features = {
        "Duration (s)": round(duration, 2),
        "Sampling Rate (Hz)": fs,
        "RMS Energy": round(float(rms_energy), 4),
        "Zero Crossing Rate": round(float(zcr), 4),
        "Mean Pitch f0 (Hz)": round(float(mean_f0), 2) if not np.isnan(mean_f0) else 0.0,
        "Pitch StdDev (Hz)": round(float(std_f0), 2) if not np.isnan(std_f0) else 0.0,
        "Jitter (%)": round(float(jitter), 2),
        "Shimmer (%)": round(float(shimmer), 2),
        "HNR (dB)": round(float(hnr), 2),
        "Formant F1 (Hz)": round(formants[0], 1) if len(formants) > 0 else 0.0,
        "Formant F2 (Hz)": round(formants[1], 1) if len(formants) > 1 else 0.0,
        "Formant F3 (Hz)": round(formants[2], 1) if len(formants) > 2 else 0.0,
    }
    for i, m in enumerate(mfcc_means):
        features[f"MFCC_{i}"] = round(float(m), 2)

    return features

# Run analysis
profile = analyze_audio_signal("voice.wav")
print("\n" + "="*40)
print("     COMPREHENSIVE VOICE PROFILE")
print("="*40)
for k, v in profile.items():
    print(f"{k:<22} : {v}")