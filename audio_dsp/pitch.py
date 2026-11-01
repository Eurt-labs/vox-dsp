"""
Fundamental Pitch (f0) Tracking & Intonation Analysis.
Uses Normalized Autocorrelation with sub-sample Parabolic Peak Interpolation
and confidence-based Voiced/Unvoiced gating.
"""

import numpy as np


def track_pitch_contour(
    signal,
    fs,
    frame_size=512,
    hop_size=160,
    fmin=75.0,
    fmax=400.0,
    voicing_threshold=0.35,
    min_energy_ratio=0.01,
):
    """
    Extracts high-resolution pitch contour (f0) across time.

    Parameters:
        signal (np.ndarray): 1D audio waveform.
        fs (int): Sampling frequency in Hz.
        frame_size (int): Analysis frame size in samples.
        hop_size (int): Hop step in samples.
        fmin (float): Minimum human pitch in Hz (default: 75 Hz).
        fmax (float): Maximum human pitch in Hz (default: 400 Hz).
        voicing_threshold (float): Minimum normalized autocorrelation peak for voicing.
        min_energy_ratio (float): Relative frame energy threshold compared to peak frame.

    Returns:
        times (np.ndarray): Timestamps for each frame center in seconds.
        pitches (np.ndarray): Fundamental frequency in Hz (NaN for unvoiced/silent frames).
        voicing_strength (np.ndarray): Normalized correlation peak magnitude in [0, 1].
    """
    min_lag = int(fs / fmax)
    max_lag = int(fs / fmin)

    num_frames = max(1, (len(signal) - frame_size) // hop_size)
    times = np.zeros(num_frames)
    pitches = np.full(num_frames, np.nan)
    voicing_strength = np.zeros(num_frames)

    # Calculate global peak frame energy for adaptive thresholding
    global_energies = [
        np.mean(signal[i * hop_size : i * hop_size + frame_size] ** 2)
        for i in range(num_frames)
    ]
    peak_energy = max(np.max(global_energies), 1e-10)

    window = np.hanning(frame_size)

    for i in range(num_frames):
        start = i * hop_size
        raw_frame = signal[start : start + frame_size]
        times[i] = (start + frame_size / 2.0) / fs

        frame_energy = np.mean(raw_frame**2)
        if frame_energy < min_energy_ratio * peak_energy:
            # Silence / ambient noise
            continue

        frame = raw_frame * window

        # Compute autocorrelation
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        r0 = corr[0]
        if r0 <= 1e-12:
            continue

        # Normalized autocorrelation values in search range
        search_range = corr[min_lag : min(max_lag + 1, len(corr))]
        if len(search_range) < 3:
            continue

        # Normalize by zero-lag energy
        norm_search = search_range / r0
        peak_idx = np.argmax(norm_search)
        peak_val = norm_search[peak_idx]
        actual_lag = min_lag + peak_idx

        voicing_strength[i] = min(1.0, max(0.0, float(peak_val)))

        if peak_val >= voicing_threshold:
            # Sub-sample parabolic interpolation around peak
            if 0 < peak_idx < len(norm_search) - 1:
                alpha = norm_search[peak_idx - 1]
                beta = norm_search[peak_idx]
                gamma = norm_search[peak_idx + 1]
                denom = 2.0 * (alpha - 2.0 * beta + gamma)
                if abs(denom) > 1e-12:
                    delta = (alpha - gamma) / denom
                    interpolated_lag = actual_lag + delta
                else:
                    interpolated_lag = float(actual_lag)
            else:
                interpolated_lag = float(actual_lag)

            if interpolated_lag > 0:
                f0 = fs / interpolated_lag
                if fmin <= f0 <= fmax:
                    pitches[i] = f0

    # Median filtering along pitch contour to remove occasional 1-frame octave glitches
    smoothed_pitches = pitches.copy()
    for i in range(1, num_frames - 1):
        window_vals = [smoothed_pitches[j] for j in range(i - 1, i + 2) if not np.isnan(smoothed_pitches[j])]
        if len(window_vals) >= 2 and np.isnan(smoothed_pitches[i]):
            smoothed_pitches[i] = np.median(window_vals)
        elif len(window_vals) == 3 and not np.isnan(smoothed_pitches[i]):
            med = np.median(window_vals)
            if abs(smoothed_pitches[i] - med) > 40.0:  # Large sudden jump
                smoothed_pitches[i] = med

    return times, smoothed_pitches, voicing_strength


def compute_pitch_statistics(pitches):
    """
    Computes statistical and prosodic metrics from a pitch contour.

    Returns:
        dict: Pitch metrics including mean, median, standard deviation, range, and voiced ratio.
    """
    valid = pitches[~np.isnan(pitches)]
    if len(valid) == 0:
        return {
            "mean_f0": 0.0,
            "median_f0": 0.0,
            "std_f0": 0.0,
            "min_f0": 0.0,
            "max_f0": 0.0,
            "pitch_range_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "voiced_ratio": 0.0,
        }

    median_f0 = float(np.median(valid))
    mean_f0 = float(np.mean(valid))
    std_f0 = float(np.std(valid))
    min_f0 = float(np.percentile(valid, 5))
    max_f0 = float(np.percentile(valid, 95))
    pitch_range_hz = max_f0 - min_f0

    semitones = 12.0 * np.log2(max_f0 / max(min_f0, 1e-5)) if min_f0 > 0 else 0.0
    voiced_ratio = len(valid) / len(pitches)

    return {
        "mean_f0": round(mean_f0, 2),
        "median_f0": round(median_f0, 2),
        "std_f0": round(std_f0, 2),
        "min_f0": round(min_f0, 2),
        "max_f0": round(max_f0, 2),
        "pitch_range_hz": round(pitch_range_hz, 2),
        "pitch_range_semitones": round(semitones, 2),
        "voiced_ratio": round(voiced_ratio, 3),
    }
