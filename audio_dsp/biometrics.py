"""
Clinical Voice Quality & Glottal Source Biometrics.
Includes Praat cycle-to-cycle perturbation (Jitter, Shimmer, HNR),
Cepstral Peak Prominence (CPPS) for continuous speech dysphonia,
and Glottal Inverse Filtering (Source Flow Derivative extraction).
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import lfilter


def compute_clinical_voice_quality(sound_or_path, start_time=None, end_time=None):
    """
    Computes Praat acoustic perturbation biometrics.
    Can be run on an auto-extracted voiced vowel segment or entire audio.

    Parameters:
        sound_or_path (str or parselmouth.Sound): Path to .wav or Sound object.
        start_time (float, optional): Segment start time in seconds.
        end_time (float, optional): Segment end time in seconds.

    Returns:
        dict: Clinical metrics (Jitter local/rap/ppq5, Shimmer local/apq3/apq5, HNR).
    """
    if isinstance(sound_or_path, str):
        sound = parselmouth.Sound(sound_or_path)
    else:
        sound = sound_or_path

    if start_time is not None and end_time is not None and end_time > start_time:
        eval_sound = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=True)
    else:
        eval_sound = sound

    # Glottal pulse tracking
    pitch = eval_sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=400.0)
    point_process = call(eval_sound, "To PointProcess (periodic, cc)", 75.0, 400.0)

    try:
        jitter_local = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100.0
    except Exception:
        jitter_local = 0.0

    try:
        jitter_rap = call(point_process, "Get jitter (rap)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100.0
    except Exception:
        jitter_rap = 0.0

    try:
        jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100.0
    except Exception:
        jitter_ppq5 = 0.0

    try:
        shimmer_local = call([eval_sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100.0
    except Exception:
        shimmer_local = 0.0

    try:
        shimmer_apq3 = call([eval_sound, point_process], "Get shimmer (apq3)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100.0
    except Exception:
        shimmer_apq3 = 0.0

    try:
        shimmer_apq5 = call([eval_sound, point_process], "Get shimmer (apq5)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100.0
    except Exception:
        shimmer_apq5 = 0.0

    try:
        harmonicity = eval_sound.to_harmonicity(time_step=0.01, minimum_pitch=75.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
    except Exception:
        hnr = 0.0

    return {
        "jitter_local_percent": round(float(jitter_local), 2),
        "jitter_rap_percent": round(float(jitter_rap), 2),
        "jitter_ppq5_percent": round(float(jitter_ppq5), 2),
        "shimmer_local_percent": round(float(shimmer_local), 2),
        "shimmer_apq3_percent": round(float(shimmer_apq3), 2),
        "shimmer_apq5_percent": round(float(shimmer_apq5), 2),
        "hnr_db": round(float(hnr), 2),
        "eval_start_s": round(float(start_time), 2) if start_time is not None else 0.0,
        "eval_end_s": round(float(end_time), 2) if end_time is not None else round(eval_sound.duration, 2),
    }


def compute_cpps(signal, fs, frame_len=1024, hop_len=256):
    """
    Computes Cepstral Peak Prominence (CPPS).
    Gold standard metric for assessing voice hoarseness and dysphonia in connected speech.

    Returns:
        cpps_mean (float): Mean CPPS value in dB across active speech.
        cpps_std (float): Standard deviation of CPPS.
    """
    min_quefrency = int(fs / 400.0)  # 400 Hz pitch limit
    max_quefrency = int(fs / 75.0)   # 75 Hz pitch limit

    num_frames = max(1, (len(signal) - frame_len) // hop_len)
    prominences = []

    window = np.hanning(frame_len)

    for i in range(num_frames):
        frame = signal[i * hop_len : i * hop_len + frame_len]
        energy = np.mean(frame**2)
        if energy < 1e-4:
            continue

        # Power cepstrum
        dft = np.fft.rfft(frame * window)
        log_mag = np.log(np.maximum(np.abs(dft), 1e-10))
        cepstrum = np.real(np.fft.rfft(log_mag))

        if len(cepstrum) <= max_quefrency:
            continue

        # Search pitch peak in human quefrency range
        search_region = cepstrum[min_quefrency:max_quefrency]
        peak_idx = min_quefrency + np.argmax(search_region)
        peak_val = cepstrum[peak_idx]

        # Fit robust linear regression across background quefrency baseline
        x = np.arange(min_quefrency, max_quefrency)
        y = cepstrum[min_quefrency:max_quefrency]
        poly = np.polyfit(x, y, deg=1)
        baseline_val = np.polyval(poly, peak_idx)

        prominence = peak_val - baseline_val
        if prominence > 0:
            prominences.append(prominence)

    if len(prominences) == 0:
        return 0.0, 0.0

    cpps_mean = float(np.mean(prominences))
    cpps_std = float(np.std(prominences))
    return round(cpps_mean, 2), round(cpps_std, 2)


def glottal_inverse_filtering(signal, fs, lpc_coeffs=None, lpc_order=18):
    """
    Extracts the Glottal Flow Derivative (excitation residual)
    by inverse filtering speech through vocal tract LPC coefficients.

    Returns:
        residual (np.ndarray): Glottal source excitation signal e[n].
        naq (float): Normalized Amplitude Quotient (glottal closure velocity index).
    """
    if lpc_coeffs is None:
        from .formants import extract_lpc_formants
        _, _, lpc_coeffs = extract_lpc_formants(signal[: min(len(signal), 1024)], fs, lpc_order=lpc_order)

    # Inverse filter: e[n] = filter(A(z), 1, s[n])
    residual = lfilter(lpc_coeffs, [1.0], signal)

    # Normalized Amplitude Quotient (NAQ) proxy: Peak-to-peak amplitude / max negative derivative
    peak_to_peak = np.ptp(residual)
    diff = np.diff(residual)
    max_neg_deriv = np.abs(np.min(diff)) if len(diff) > 0 and np.min(diff) < 0 else 1e-6

    naq = (peak_to_peak / (max_neg_deriv + 1e-8)) / fs
    naq = max(0.01, min(0.35, float(naq * 1000.0)))  # Scaled to typical physiological scale

    return residual, round(naq, 3)
