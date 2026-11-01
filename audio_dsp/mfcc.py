"""
Mel-Frequency Cepstral Coefficients (MFCCs) with CMVN & Dynamic Derivatives.
Extracts 13 cepstral coefficients, applies Cepstral Mean and Variance Normalization (CMVN),
and computes Delta (velocity) and Delta-Delta (acceleration) features.
"""

import numpy as np
import librosa


def extract_mfcc_features(
    signal,
    fs,
    n_mfcc=13,
    n_fft=512,
    hop_len=160,
    use_cmvn=True,
    vad_labels=None,
):
    """
    Extracts comprehensive perceptual cepstral feature representation.

    Parameters:
        signal (np.ndarray): 1D audio waveform.
        fs (int): Sampling frequency in Hz.
        n_mfcc (int): Number of coefficients (default: 13, c0 to c12).
        n_fft (int): Window length for STFT mel filterbanks.
        hop_len (int): Frame shift in samples (10 ms at 16kHz).
        use_cmvn (bool): Apply Cepstral Mean & Variance Normalization.
        vad_labels (np.ndarray, optional): Array of VAD labels to filter silent frames from summary stats.

    Returns:
        mfccs_raw (np.ndarray): Shape (n_mfcc, T) raw coefficients.
        mfccs_norm (np.ndarray): Shape (n_mfcc, T) normalized coefficients.
        deltas (np.ndarray): Shape (n_mfcc, T) velocity features.
        delta2 (np.ndarray): Shape (n_mfcc, T) acceleration features.
        summary_dict (dict): Aggregated biometric cepstral statistics.
        times (np.ndarray): Time vector in seconds.
    """
    # 1. Compute raw MFCCs
    mfccs_raw = librosa.feature.mfcc(
        y=signal,
        sr=fs,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_len,
    )
    times = librosa.times_like(mfccs_raw[0], sr=fs, hop_length=hop_len)

    # 2. Compute dynamic derivatives (Delta and Delta-Delta)
    deltas = librosa.feature.delta(mfccs_raw, order=1)
    delta2 = librosa.feature.delta(mfccs_raw, order=2)

    # 3. Apply CMVN (Cepstral Mean and Variance Normalization)
    mfccs_norm = mfccs_raw.copy()
    if use_cmvn:
        means = np.mean(mfccs_raw, axis=1, keepdims=True)
        stds = np.std(mfccs_raw, axis=1, keepdims=True) + 1e-8
        mfccs_norm = (mfccs_raw - means) / stds

    # 4. Generate summary dictionary (masking silence if VAD labels provided)
    if vad_labels is not None and len(vad_labels) == mfccs_raw.shape[1]:
        speech_mask = vad_labels != "silence"
        if not np.any(speech_mask):
            speech_mask = np.ones(mfccs_raw.shape[1], dtype=bool)
    else:
        speech_mask = np.ones(mfccs_raw.shape[1], dtype=bool)

    active_raw = mfccs_raw[:, speech_mask]
    active_deltas = deltas[:, speech_mask]

    summary = {}
    for i in range(n_mfcc):
        summary[f"MFCC_{i}_mean"] = round(float(np.mean(active_raw[i])), 2)
        summary[f"MFCC_{i}_std"] = round(float(np.std(active_raw[i])), 2)
        summary[f"MFCC_delta_{i}_mean"] = round(float(np.mean(active_deltas[i])), 2)

    return mfccs_raw, mfccs_norm, deltas, delta2, summary, times

