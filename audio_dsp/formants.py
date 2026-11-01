"""
Vocal Tract Filter Modeling & Formant Tracking via Linear Predictive Coding (LPC).
Includes Pre-Emphasis, Adaptive Model Order, 3-dB Bandwidth Pruning,
Physical Vocal Tract Length (VTL) Estimation, and IPA Vowel Space Mapping.
"""

import numpy as np
from scipy.linalg import solve_toeplitz


def apply_pre_emphasis(signal, alpha=0.97):
    """
    Applies a +6 dB/octave first-order high-pass pre-emphasis filter:
    y[n] = x[n] - alpha * x[n-1]
    """
    if len(signal) == 0:
        return signal
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def estimate_vocal_tract_length(f0_median, f1, f2, f3):
    """
    Estimates the speaker's physical Vocal Tract Length (VTL in cm)
    and demographic class using pitch and formant dispersion.

    Parameters:
        f0_median (float): Median pitch in Hz.
        f1, f2, f3 (float): Formant frequencies in Hz.

    Returns:
        vtl_cm (float): Estimated vocal tract length in cm.
        demographic (str): Inferred speaker class ('Adult Male', 'Adult Female', 'Child').
        recommended_lpc_order (int): Optimal LPC filter order for this speaker.
    """
    c = 34300.0  # Speed of sound in warm moist air in cm/s

    # 1. Primary biological classification by glottal fundamental frequency (f0)
    if f0_median > 0 and f0_median <= 155.0:
        demographic = "Adult Male"
        prior_vtl = 17.2
        min_vtl, max_vtl = 15.0, 19.5
    elif f0_median >= 240.0:
        demographic = "Child / High Pitch"
        prior_vtl = 11.5
        min_vtl, max_vtl = 9.5, 13.0
    else:
        demographic = "Adult Female"
        prior_vtl = 14.5
        min_vtl, max_vtl = 13.0, 16.0

    # 2. Formant-based VTL refinement via Fitch's method
    valid_formants = [f for f in [f1, f2, f3] if f > 200.0]
    if len(valid_formants) >= 2:
        # Fitch tube model: L_n = (2n - 1) * c / (4 * F_n)
        lengths = [
            ((2 * n - 1) * c) / (4.0 * fn)
            for n, fn in enumerate(valid_formants[:3], start=1)
        ]
        fitch_vtl = np.mean(lengths)
        # Weighted blend between pitch demographic prior and formant estimate
        vtl_cm = 0.5 * prior_vtl + 0.5 * fitch_vtl
        vtl_cm = max(min_vtl, min(max_vtl, float(vtl_cm)))
    else:
        vtl_cm = prior_vtl

    # 3. Dynamic LPC filter order: p = 2 * (L / c) * fs + 2
    lpc_order = int(round(2.0 * (vtl_cm / c) * 16000.0 + 2))
    lpc_order = max(14, min(20, lpc_order))

    return round(vtl_cm, 1), demographic, lpc_order


def extract_lpc_formants(
    frame,
    fs,
    lpc_order=18,
    pre_emphasis=0.97,
    max_bandwidth=900.0,
):
    """
    Extracts vowel formant resonances (F1, F2, F3) using pre-emphasized LPC
    with robust pole-radius and bandwidth filtering.

    Parameters:
        frame (np.ndarray): 1D audio frame (typically 25-35 ms).
        fs (int): Sampling frequency in Hz.
        lpc_order (int): Filter order (default: 18 for 16 kHz).
        pre_emphasis (float): Pre-emphasis coefficient (default: 0.97).
        max_bandwidth (float): Maximum allowable 3-dB bandwidth in Hz (default: 900 Hz).

    Returns:
        formants (list of float): Sorted formant frequencies in Hz.
        bandwidths (list of float): Formant bandwidths in Hz.
        a (np.ndarray): LPC polynomial coefficients [1, a1, a2, ..., ap].
    """
    if lpc_order is None or lpc_order < 12:
        lpc_order = 18

    # 1. Pre-emphasis filter
    if pre_emphasis > 0:
        emp_frame = apply_pre_emphasis(frame, alpha=pre_emphasis)
    else:
        emp_frame = frame

    # 2. Windowing
    windowed = emp_frame * np.hamming(len(emp_frame))

    # 3. Autocorrelation & Toeplitz solver
    n = len(windowed)
    r = np.correlate(windowed, windowed, mode="full")[n - 1 :]
    r0 = r[0]
    if r0 < 1e-12:
        return [], [], np.array([1.0])

    r_norm = r / r0
    a = np.zeros(lpc_order + 1)
    a[0] = 1.0

    try:
        a[1:] = solve_toeplitz((r_norm[:lpc_order], r_norm[:lpc_order]), -r_norm[1 : lpc_order + 1])
    except Exception:
        return [], [], a

    # 4. Polynomial Root Finding
    roots = np.roots(a)

    # 5. Extract frequencies and bandwidths
    valid_formants = []
    valid_bandwidths = []

    for root in roots:
        if np.imag(root) > 0.01:
            freq = np.arctan2(np.imag(root), np.real(root)) * (fs / (2.0 * np.pi))
            radius = np.abs(root)

            # Bandwidth: B = - (fs / pi) * ln(|r|)
            if 0.0 < radius < 1.0:
                bw = -(fs / np.pi) * np.log(radius)
            else:
                bw = 9999.0

            # Filter valid vowel formants:
            # - Frequency in speech range [200 Hz, 4000 Hz]
            # - Radius >= 0.75 (bandwidth <= 900 Hz) to avoid spurious spectral tilt poles
            if 200.0 < freq < min(4000.0, fs / 2.0 - 200.0) and radius >= 0.75 and bw <= max_bandwidth:
                valid_formants.append(freq)
                valid_bandwidths.append(bw)

    # Sort formants by ascending frequency
    sorted_pairs = sorted(zip(valid_formants, valid_bandwidths), key=lambda x: x[0])
    formants = [round(float(f), 1) for f, _ in sorted_pairs]
    bandwidths = [round(float(b), 1) for _, b in sorted_pairs]

    return formants, bandwidths, a


def track_formants_contour(signal, fs, vad_labels=None, frame_len=512, hop_len=160, lpc_order=18):
    """
    Tracks dynamic formant tracks F1(t), F2(t), F3(t) across voiced regions.

    Returns:
        times (np.ndarray): Timestamps in seconds.
        f1_track, f2_track, f3_track (np.ndarray): Formant curves in Hz (NaN for unvoiced).
    """
    num_frames = max(1, (len(signal) - frame_len) // hop_len)
    times = np.zeros(num_frames)
    f1_track = np.full(num_frames, np.nan)
    f2_track = np.full(num_frames, np.nan)
    f3_track = np.full(num_frames, np.nan)

    for i in range(num_frames):
        start = i * hop_len
        times[i] = (start + frame_len / 2.0) / fs

        if vad_labels is not None and i < len(vad_labels) and vad_labels[i] != "voiced":
            continue

        frame = signal[start : start + frame_len]
        formants, _, _ = extract_lpc_formants(frame, fs, lpc_order=lpc_order)

        if len(formants) >= 1:
            f1_track[i] = formants[0]
        if len(formants) >= 2:
            f2_track[i] = formants[1]
        if len(formants) >= 3:
            f3_track[i] = formants[2]

    return times, f1_track, f2_track, f3_track


def compute_vowel_space(f1_vals, f2_vals):
    """
    Computes Vowel Space Area (VSA) and classifies dominant vowel quadrant
    relative to International Phonetic Alphabet (IPA) standards.

    Parameters:
        f1_vals (list or np.ndarray): Valid F1 measurements.
        f2_vals (list or np.ndarray): Valid F2 measurements.

    Returns:
        dict: VSA metrics and dominant phonetic vowel description.
    """
    valid_f1 = [f for f in f1_vals if not np.isnan(f) and f > 200.0]
    valid_f2 = [f for f in f2_vals if not np.isnan(f) and f > 500.0]

    if len(valid_f1) < 4 or len(valid_f2) < 4:
        return {
            "vowel_space_area_hz2": 0.0,
            "vsa_category": "Insufficient Voiced Vowels",
            "dominant_vowel": "Unknown / Unvoiced",
        }

    f1_min, f1_max = np.percentile(valid_f1, 10), np.percentile(valid_f1, 90)
    f2_min, f2_max = np.percentile(valid_f2, 10), np.percentile(valid_f2, 90)

    vsa = 0.5 * (f1_max - f1_min) * (f2_max - f2_min)

    f1_med = np.median(valid_f1)
    f2_med = np.median(valid_f2)

    if f1_med < 400.0:
        jaw = "Close / High"
    elif f1_med < 650.0:
        jaw = "Mid / Central"
    else:
        jaw = "Open / Low"

    if f2_med > 1800.0:
        tongue = "Front (/i/, /e/, /ae/)"
    elif f2_med > 1300.0:
        tongue = "Central (/uh/, /ah/)"
    else:
        tongue = "Back (/u/, /o/, /aa/)"

    vowel_desc = f"{jaw} {tongue}"

    return {
        "vowel_space_area_hz2": round(float(vsa), 1),
        "dominant_vowel": vowel_desc,
        "f1_centroid": round(float(f1_med), 1),
        "f2_centroid": round(float(f2_med), 1),
    }
