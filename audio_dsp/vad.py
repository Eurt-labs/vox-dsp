"""
Adaptive Voice Activity Detection (VAD) & Phonetic Segmentation.
Combines Log Energy, Zero Crossing Rate (ZCR), and Spectral Entropy to classify
frames into Silence, Unvoiced Consonants, and Voiced Speech (Vowels).
"""

import numpy as np


def compute_vad_segments(signal, fs, frame_len=512, hop_len=160):
    """
    Computes time-aligned VAD classification for audio signal.

    Parameters:
        signal (np.ndarray): 1D normalized audio signal in [-1.0, 1.0].
        fs (int): Sampling frequency in Hz.
        frame_len (int): Frame length in samples (default: 512 = 32ms at 16kHz).
        hop_len (int): Step between frames in samples (default: 160 = 10ms at 16kHz).

    Returns:
        times (np.ndarray): Frame center time stamps in seconds.
        labels (np.ndarray): Array of string labels ('silence', 'unvoiced', 'voiced').
        energies_db (np.ndarray): Frame energy in dB.
        zcr (np.ndarray): Zero crossing rate per frame.
    """
    num_frames = max(1, (len(signal) - frame_len) // hop_len)
    times = np.zeros(num_frames)
    energies = np.zeros(num_frames)
    zcr = np.zeros(num_frames)
    spectral_entropy = np.zeros(num_frames)

    window = np.hanning(frame_len)

    for i in range(num_frames):
        start = i * hop_len
        frame = signal[start : start + frame_len]
        times[i] = (start + frame_len / 2.0) / fs

        # 1. Short-Time Energy
        frame_e = np.mean(frame**2)
        energies[i] = frame_e

        # 2. Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame) > 0)))
        zcr[i] = zero_crossings / (frame_len - 1)

        # 3. Spectral Entropy (harmonicity indicator)
        windowed = frame * window
        fft_mag = np.abs(np.fft.rfft(windowed))
        p = fft_mag / (np.sum(fft_mag) + 1e-12)
        p = p[p > 1e-12]
        # Normalized entropy in [0, 1]
        entropy = -np.sum(p * np.log2(p)) / np.log2(len(fft_mag))
        spectral_entropy[i] = entropy

    # Convert energy to dB
    max_e = np.max(energies) if np.max(energies) > 1e-12 else 1e-12
    energies_db = 10.0 * np.log10(np.maximum(energies, 1e-12) / max_e)

    # Dynamic noise floor estimation (15th percentile of energy)
    noise_floor_db = np.percentile(energies_db, 20)
    speech_threshold_db = max(noise_floor_db + 15.0, -40.0)

    labels = np.array(["silence"] * num_frames, dtype=object)

    for i in range(num_frames):
        if energies_db[i] < speech_threshold_db:
            labels[i] = "silence"
        else:
            # High ZCR and high entropy characterize unvoiced fricatives/plosives
            # Low ZCR, lower entropy, and high energy characterize periodic voiced vowels
            if zcr[i] > 0.18 or spectral_entropy[i] > 0.85:
                labels[i] = "unvoiced"
            else:
                labels[i] = "voiced"

    # Smooth short transient dropouts (morphological closing: 3-frame window)
    for i in range(1, num_frames - 1):
        if labels[i - 1] == "voiced" and labels[i + 1] == "voiced" and labels[i] != "voiced":
            labels[i] = "voiced"

    return times, labels, energies_db, zcr


def extract_longest_voiced_segment(signal, fs, min_duration=0.35, frame_len=512, hop_len=160):
    """
    Finds the longest stable voiced speech segment (e.g. sustained vowel)
    for high-precision clinical perturbation and LPC analysis.

    Returns:
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
        vowel_segment (np.ndarray): Audio slice containing the stable vowel.
    """
    times, labels, _, _ = compute_vad_segments(signal, fs, frame_len, hop_len)
    is_voiced = labels == "voiced"

    longest_start = 0
    longest_len = 0
    cur_start = None
    cur_len = 0

    for idx, voiced in enumerate(is_voiced):
        if voiced:
            if cur_start is None:
                cur_start = idx
                cur_len = 1
            else:
                cur_len += 1
        else:
            if cur_start is not None:
                if cur_len > longest_len:
                    longest_len = cur_len
                    longest_start = cur_start
                cur_start = None
                cur_len = 0

    if cur_start is not None and cur_len > longest_len:
        longest_len = cur_len
        longest_start = cur_start

    dt = hop_len / fs
    duration = longest_len * dt

    if duration >= min_duration and longest_len > 0:
        start_time = times[longest_start] - (frame_len / (2 * fs))
        end_time = times[longest_start + longest_len - 1] + (frame_len / (2 * fs))
        start_time = max(0.0, start_time)
        end_time = min(len(signal) / fs, end_time)
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        return start_time, end_time, signal[start_sample:end_sample]

    # Fallback if no clean contiguous segment >= min_duration:
    # Use middle 1-second segment or entire clip
    clip_dur = len(signal) / fs
    mid = clip_dur / 2.0
    start_time = max(0.0, mid - 0.5)
    end_time = min(clip_dur, start_time + 1.0)
    return start_time, end_time, signal[int(start_time * fs) : int(end_time * fs)]
