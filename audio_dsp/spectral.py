"""
Fourier Spectral Decomposition & STFT Spectrogram.
Computes Short-Time Fourier Transforms, narrowband magnitude spectrograms,
and single-frame DFT harmonic spectra for active speech frames.
"""

import numpy as np
from scipy.signal import spectrogram


def compute_stft_spectrogram(
    signal,
    fs,
    n_fft=512,
    hop_len=128,
    window="hann",
    fmax=4000.0,
):
    """
    Computes smooth STFT Spectrogram matrix with 75% overlap.

    Returns:
        times (np.ndarray): Time vector in seconds.
        freqs (np.ndarray): Frequency vector in Hz (limited to 0..fmax).
        spectrogram_db (np.ndarray): 2D energy matrix in decibels.
    """
    noverlap = n_fft - hop_len
    frequencies, times, Sxx = spectrogram(
        signal,
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=noverlap,
        scaling="spectrum",
    )

    # Convert to dB
    spectrogram_db = 10.0 * np.log10(np.maximum(Sxx, 1e-12))
    # Normalize relative to clip peak
    spectrogram_db -= np.max(spectrogram_db)

    # Filter frequencies up to fmax
    freq_mask = frequencies <= fmax
    frequencies = frequencies[freq_mask]
    spectrogram_db = spectrogram_db[freq_mask, :]

    return times, frequencies, spectrogram_db


def compute_active_frame_dft(signal, fs, frame_len=512, target_time=None):
    """
    Selects a high-energy stationary speech frame and computes its Hann-windowed DFT spectrum.

    Parameters:
        signal (np.ndarray): 1D audio waveform.
        fs (int): Sampling frequency in Hz.
        frame_len (int): Frame length in samples.
        target_time (float, optional): Exact time in seconds to slice. If None, selects peak energy frame.

    Returns:
        freqs (np.ndarray): Frequency bins in Hz.
        mag_db (np.ndarray): Magnitude spectrum in dB.
        peak_freq (float): Frequency of dominant spectral peak.
        peak_db (float): Magnitude of dominant peak in dB.
        frame_time (float): Center time of the analyzed frame in seconds.
    """
    if target_time is not None:
        start_idx = max(0, min(len(signal) - frame_len, int(target_time * fs - frame_len / 2)))
    else:
        # Search loudest 32ms frame
        hop = frame_len // 4
        num_steps = (len(signal) - frame_len) // hop
        energies = [
            np.sum(signal[i * hop : i * hop + frame_len] ** 2)
            for i in range(num_steps)
        ]
        start_idx = np.argmax(energies) * hop if len(energies) > 0 else 0

    frame = signal[start_idx : start_idx + frame_len]
    frame_time = (start_idx + frame_len / 2.0) / fs

    window = np.hanning(frame_len)
    windowed = frame * window

    dft = np.fft.rfft(windowed, n=1024)
    mag = np.abs(dft) / frame_len
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-10))
    freqs = np.fft.rfftfreq(1024, d=1.0 / fs)

    # Search dominant peak in speech range (80 Hz to 3500 Hz)
    speech_mask = (freqs >= 80.0) & (freqs <= 3500.0)
    sub_indices = np.where(speech_mask)[0]
    peak_local = np.argmax(mag_db[speech_mask])
    peak_bin = sub_indices[peak_local]

    peak_freq = float(freqs[peak_bin])
    peak_db = float(mag_db[peak_bin])

    return freqs, mag_db, peak_freq, peak_db, round(frame_time, 2)

