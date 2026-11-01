"""
vox-dsp Diagnostic Visualizer & Multi-Plot Engine.
Provides standalone, publication-grade individual plot renderers:
1. Active Frame Discrete Fourier Transform (DFT) Magnitude Spectrum (with raw CSV export)
2. STFT Time-Frequency Spectrogram (0-4000 Hz)
3. Fundamental Pitch (f0) Intonation Tracking Contour
4. Pre-Emphasized LPC Vocal Tract Filter Envelope vs FFT Harmonics
5. CMVN-Normalized Mel-Frequency Cepstral Coefficients (MFCCs)
6. Phonetic IPA Vowel Space Area (F1 vs F2 Quadrilateral)
7. Glottal Flow Derivative Residual Waveform (Inverse Filtered Source)
8. Unified 6-Panel Composite Diagnostic Dashboard
9. Complete ZIP Bundle Exporter for all plots and raw data
"""

import os
import csv
import zipfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from audio_dsp.vad import compute_vad_segments, extract_longest_voiced_segment
from audio_dsp.pitch import track_pitch_contour, compute_pitch_statistics
from audio_dsp.formants import (
    extract_lpc_formants,
    track_formants_contour,
    estimate_vocal_tract_length,
    compute_vowel_space,
)
from audio_dsp.spectral import compute_stft_spectrogram, compute_active_frame_dft
from audio_dsp.mfcc import extract_mfcc_features
from audio_dsp.biometrics import glottal_inverse_filtering


def setup_dark_style(fig, axes):
    """Applies a clean, high-contrast dark zinc aesthetic to matplotlib figures."""
    fig.patch.set_facecolor('#09090b')
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax in axes:
        ax.set_facecolor('#121215')
        ax.tick_params(colors='#a1a1aa', labelsize=8)
        ax.xaxis.label.set_color('#d4d4d8')
        ax.yaxis.label.set_color('#d4d4d8')
        ax.title.set_color('#f4f4f5')
        for spine in ax.spines.values():
            spine.set_color('#27272a')
        ax.grid(True, color='#27272a', linestyle='--', linewidth=0.7, alpha=0.7)


# -------------------------------------------------------------
# 1. STANDALONE DFT SPECTRUM PLOT
# -------------------------------------------------------------
def render_dft_plot(audio_file, save_path="dft_spectrum.png", csv_save_path=None):
    """
    Renders the Discrete Fourier Transform (DFT) magnitude spectrum of the active speech frame
    with labeled dominant peak frequency, harmonics grid, and raw data export.
    """
    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    vowel_start, vowel_end, _ = extract_longest_voiced_segment(signal, fs)
    target_time = (vowel_start + vowel_end) / 2.0

    freqs, mag_db, peak_freq, peak_db, frame_time = compute_active_frame_dft(
        signal, fs, target_time=target_time
    )

    # Export raw DFT points to CSV if requested
    if csv_save_path:
        with open(csv_save_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frequency_Hz", "Magnitude_dB"])
            for f_val, m_val in zip(freqs, mag_db):
                writer.writerow([round(float(f_val), 2), round(float(m_val), 2)])

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    setup_dark_style(fig, ax)

    ax.plot(freqs, mag_db, color="#38bdf8", linewidth=1.2, label="DFT Magnitude Spectrum (Hann)")
    ax.axvline(peak_freq, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Dominant Peak: {peak_freq:.1f} Hz ({peak_db:.1f} dB)")

    # Mark first 4 harmonics of dominant peak if applicable
    if 70 < peak_freq < 400:
        for h in range(2, 6):
            h_freq = peak_freq * h
            if h_freq <= 4000:
                ax.axvline(h_freq, color="#52525b", linestyle=":", linewidth=0.9, alpha=0.8)

    ax.set_xlim(0, 4000)
    ax.set_ylim(max(np.min(mag_db), -90), max(np.max(mag_db) + 5, 0))
    ax.set_title(f"Discrete Fourier Transform (DFT) Magnitude Spectrum — Active Frame at t={frame_time}s", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized Magnitude (dB)")
    ax.legend(loc="upper right", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 2. STANDALONE STFT SPECTROGRAM PLOT
# -------------------------------------------------------------
def render_spectrogram_plot(audio_file, save_path="stft_spectrogram.png"):
    """
    Renders high-resolution STFT Time-Frequency Spectrogram (0-4000 Hz).
    """
    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    times_spec, freqs_spec, S_db = compute_stft_spectrogram(signal, fs, fmax=4000.0)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    setup_dark_style(fig, ax)

    pcm = ax.pcolormesh(times_spec, freqs_spec, S_db, shading="gouraud", cmap="magma", vmin=-60, vmax=0)
    ax.set_ylim(0, 4000)
    ax.set_xlim(0, len(signal) / fs)
    ax.set_title("Short-Time Fourier Transform (STFT) Speech Spectrogram (0 - 4000 Hz)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("Magnitude (dB)", color="#d4d4d8", fontsize=8)
    cb.ax.tick_params(colors="#a1a1aa", labelsize=8)
    cb.outline.set_edgecolor("#27272a")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 3. STANDALONE PITCH CONTOUR PLOT
# -------------------------------------------------------------
def render_pitch_plot(audio_file, save_path="pitch_contour.png"):
    """
    Renders Fundamental Pitch (f0) Intonation Tracking Contour.
    """
    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    times_pitch, pitches, _ = track_pitch_contour(signal, fs)
    pitch_stats = compute_pitch_statistics(pitches)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    setup_dark_style(fig, ax)

    ax.plot(times_pitch, pitches, color="#f43f5e", linewidth=1.8, label="Pitch Contour (f0)")

    med_f0 = pitch_stats.get("median_f0", 0.0)
    std_f0 = pitch_stats.get("std_f0", 0.0)
    if med_f0 > 0:
        ax.axhline(med_f0, color="#38bdf8", linestyle="--", linewidth=1.2, label=f"Median f0: {med_f0:.1f} Hz")
        ax.axhspan(max(50, med_f0 - std_f0), med_f0 + std_f0, color="#0284c7", alpha=0.15, label=f"±1σ Spread: {pitch_stats.get('min_f0', 0):.0f} - {pitch_stats.get('max_f0', 0):.0f} Hz ({pitch_stats.get('pitch_range_semitones', 0):.1f} st)")

    ax.set_xlim(0, len(signal) / fs)
    ax.set_ylim(50, 350)
    ax.set_title("Glottal Fundamental Frequency (f0) Intonation Contour Over Time", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch (Hz)")
    ax.legend(loc="upper right", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 4. STANDALONE LPC FORMANTS PLOT
# -------------------------------------------------------------
def render_lpc_plot(audio_file, save_path="lpc_formants.png"):
    """
    Renders pre-emphasized LPC Vocal Tract Filter Envelope vs FFT Harmonics with labeled formants.
    """
    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    vowel_start, vowel_end, _ = extract_longest_voiced_segment(signal, fs)
    target_time = (vowel_start + vowel_end) / 2.0

    frame_len = int(0.03 * fs)
    start_idx = max(0, min(len(signal) - frame_len, int(target_time * fs - frame_len / 2)))
    vowel_frame = signal[start_idx : start_idx + frame_len]

    formants, bandwidths, lpc_a = extract_lpc_formants(vowel_frame, fs, lpc_order=18, pre_emphasis=0.97, max_bandwidth=900.0)

    windowed = vowel_frame * np.hamming(len(vowel_frame))
    freqs_dft = np.fft.rfftfreq(1024, 1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(windowed, 1024))
    fft_db = 20.0 * np.log10(np.maximum(fft_mag, 1e-5))

    w, h_lpc = np.fft.rfftfreq(1024, 1.0 / fs), 1.0 / np.fft.rfft(lpc_a, 1024)
    lpc_db = 20.0 * np.log10(np.maximum(np.abs(h_lpc), 1e-5))
    lpc_db += np.max(fft_db) - np.max(lpc_db)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    setup_dark_style(fig, ax)

    ax.plot(freqs_dft, fft_db, color="#71717a", alpha=0.5, linewidth=1.0, label="FFT Harmonics")
    ax.plot(w, lpc_db, color="#38bdf8", linewidth=2.0, label="LPC Vocal Tract Envelope (p=18)")

    colors = ["#ef4444", "#fbbf24", "#a855f7"]
    for i, f in enumerate(formants[:3]):
        c = colors[i % len(colors)]
        ax.axvline(f, color=c, linestyle="--", linewidth=1.5, label=f"Formant F{i+1}: {f:.0f} Hz (BW: {bandwidths[i]:.0f} Hz)")

    ax.set_xlim(0, 4000)
    ax.set_title(f"Linear Predictive Coding (LPC) Vocal Tract Filter Envelope at t={target_time:.2f}s", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend(loc="upper right", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 5. STANDALONE MFCC & CMVN PLOT
# -------------------------------------------------------------
def render_mfcc_plot(audio_file, save_path="mfcc_cmvn.png"):
    """
    Renders 13 Mel-Frequency Cepstral Coefficients with CMVN and c0 log energy.
    """
    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    mfccs_raw, mfccs_norm, _, _, _, times_mfcc = extract_mfcc_features(signal, fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True, constrained_layout=True)
    setup_dark_style(fig, [ax1, ax2])

    img = librosa.display.specshow(mfccs_norm[1:], x_axis="time", sr=fs, hop_length=160, cmap="coolwarm", ax=ax1, vmin=-2.5, vmax=2.5)
    ax1.set_title("Normalized Mel-Frequency Cepstral Coefficients (c1 - c12, CMVN)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MFCC Index")
    ax1.set_yticks(range(0, 12))
    ax1.set_yticklabels([f"c{i}" for i in range(1, 13)])
    cb = fig.colorbar(img, ax=ax1, pad=0.02)
    cb.set_label("Z-Score", color="#d4d4d8", fontsize=8)
    cb.ax.tick_params(colors="#a1a1aa", labelsize=8)
    cb.outline.set_edgecolor("#27272a")

    ax2.plot(times_mfcc, mfccs_raw[0], color="#38bdf8", linewidth=1.2)
    ax2.set_title("Frame Energy Profile (c0 Log Energy)", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("c0 Energy")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 6. STANDALONE PHONETIC VOWEL SPACE PLOT
# -------------------------------------------------------------
def render_vowel_space_plot(audio_file, save_path="vowel_space.png"):
    """
    Renders Phonetic Vowel Quadrilateral (F1 vs F2 IPA Acoustic Space).
    """
    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    times_vad, vad_labels, _, _ = compute_vad_segments(signal, fs)
    _, f1_tr, f2_tr, _ = track_formants_contour(signal, fs, vad_labels=vad_labels, lpc_order=18)
    vowel_space = compute_vowel_space(f1_tr, f2_tr)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    setup_dark_style(fig, ax)

    valid_mask = (~np.isnan(f1_tr)) & (~np.isnan(f2_tr)) & (f1_tr > 200) & (f2_tr > 500)
    if np.any(valid_mask):
        ax.scatter(f2_tr[valid_mask], f1_tr[valid_mask], color="#38bdf8", alpha=0.35, s=25, label="Voiced Frame Formants")

    ipa_vowels = {
        "/i/ (fleece)": (2300, 280),
        "/e/ (face)": (2000, 450),
        "/æ/ (trap)": (1750, 750),
        "/ɑ/ (palm)": (1150, 750),
        "/ɔ/ (thought)": (950, 550),
        "/u/ (goose)": (850, 300),
        "/ʌ/ (strut)": (1250, 650),
    }
    for symbol, (f2_ref, f1_ref) in ipa_vowels.items():
        ax.scatter(f2_ref, f1_ref, color="#f43f5e", marker="x", s=50)
        ax.text(f2_ref + 25, f1_ref + 15, symbol, color="#fda4af", fontsize=8, fontweight="bold")

    if "f1_centroid" in vowel_space and vowel_space["f1_centroid"] > 0:
        ax.scatter(vowel_space["f2_centroid"], vowel_space["f1_centroid"], color="#10b981", edgecolors="#ffffff", s=130, marker="*", label=f"Speaker Centroid ({vowel_space['dominant_vowel']})", zorder=5)

    ax.set_xlim(2600, 700)
    ax.set_ylim(900, 200)
    ax.set_title(f"Phonetic Vowel Space Quadrilateral (VSA: {vowel_space.get('vowel_space_area_hz2', 0):,.0f} Hz²)", fontsize=11, fontweight="bold")
    ax.set_xlabel("F2 Frequency (Hz) ← Back vs Front Tongue →")
    ax.set_ylabel("F1 Frequency (Hz) ← Open vs Close Jaw →")
    ax.legend(loc="upper left", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 7. STANDALONE GLOTTAL FLOW RESIDUAL PLOT
# -------------------------------------------------------------
def render_glottal_plot(audio_file, save_path="glottal_residual.png"):
    """
    Renders Speech Waveform vs Glottal Flow Derivative Residual (Inverse Filtered Source).
    """
    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    residual, naq = glottal_inverse_filtering(signal, fs, lpc_order=18)
    times = np.linspace(0, len(signal) / fs, len(signal))

    # Zoom into a 150 ms clean voiced segment for glottal pulse inspection
    vowel_start, vowel_end, _ = extract_longest_voiced_segment(signal, fs)
    t_start = max(0, (vowel_start + vowel_end) / 2.0 - 0.075)
    t_end = min(len(signal) / fs, t_start + 0.150)
    idx_start, idx_end = int(t_start * fs), int(t_end * fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
    setup_dark_style(fig, [ax1, ax2])

    ax1.plot(times[idx_start:idx_end], signal[idx_start:idx_end], color="#38bdf8", linewidth=1.2)
    ax1.set_title("Speech Pressure Waveform s[n] (Acoustic Output)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Amplitude")

    ax2.plot(times[idx_start:idx_end], residual[idx_start:idx_end], color="#f43f5e", linewidth=1.2)
    ax2.set_title(f"Glottal Flow Derivative Residual e[n] (Vocal Fold Excitation Source, NAQ: {naq:.3f})", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Residual")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 8. UNIFIED 6-PANEL COMPOSITE DASHBOARD
# -------------------------------------------------------------
def render_voice_dashboard(audio_file="voice.wav", save_path="voice_dashboard.png", show=True):
    """
    Renders the unified 6-panel composite dashboard.
    """
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(signal) / fs
    times_wav = np.linspace(0, duration, len(signal))

    times_vad, vad_labels, _, _ = compute_vad_segments(signal, fs)
    vowel_start, vowel_end, _ = extract_longest_voiced_segment(signal, fs)
    times_pitch, pitches, _ = track_pitch_contour(signal, fs)
    pitch_stats = compute_pitch_statistics(pitches)

    target_time = (vowel_start + vowel_end) / 2.0
    frame_len = int(0.03 * fs)
    start_idx = max(0, min(len(signal) - frame_len, int(target_time * fs - frame_len / 2)))
    vowel_frame = signal[start_idx : start_idx + frame_len]

    prelim_formants, _, _ = extract_lpc_formants(vowel_frame, fs, lpc_order=18)
    f1_init = prelim_formants[0] if len(prelim_formants) > 0 else 500.0
    f2_init = prelim_formants[1] if len(prelim_formants) > 1 else 1500.0
    f3_init = prelim_formants[2] if len(prelim_formants) > 2 else 2500.0
    vtl_cm, demographic, opt_order = estimate_vocal_tract_length(pitch_stats["median_f0"], f1_init, f2_init, f3_init)

    formants, bandwidths, lpc_a = extract_lpc_formants(vowel_frame, fs, lpc_order=opt_order, pre_emphasis=0.97, max_bandwidth=900.0)
    times_formant, f1_tr, f2_tr, f3_tr = track_formants_contour(signal, fs, vad_labels=vad_labels, lpc_order=opt_order)
    vowel_space = compute_vowel_space(f1_tr, f2_tr)
    times_spec, freqs_spec, S_db = compute_stft_spectrogram(signal, fs, fmax=4000.0)
    mfccs_raw, mfccs_norm, _, _, _, times_mfcc = extract_mfcc_features(signal, fs, vad_labels=vad_labels)

    fig, axes = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True)
    setup_dark_style(fig, axes)
    fig.suptitle(f"vox-dsp Unified Acoustic Diagnostic Dashboard: {os.path.basename(audio_file)} ({demographic}, VTL: {vtl_cm} cm)", fontsize=15, fontweight="bold", y=1.01, color="#f4f4f5")

    # Panel 1: Waveform & VAD
    ax1 = axes[0, 0]
    ax1.plot(times_wav, signal, color="#38bdf8", alpha=0.7, linewidth=0.7)
    dt = times_vad[1] - times_vad[0] if len(times_vad) > 1 else 0.01
    for t, label in zip(times_vad, vad_labels):
        if label == "voiced":
            ax1.axvspan(t - dt / 2, t + dt / 2, color="#10b981", alpha=0.15)
        elif label == "unvoiced":
            ax1.axvspan(t - dt / 2, t + dt / 2, color="#f59e0b", alpha=0.15)
    ax1.axvspan(vowel_start, vowel_end, color="#ef4444", alpha=0.25, label=f"Clinical Vowel ({vowel_start:.2f}s - {vowel_end:.2f}s)")
    ax1.set_title("1. Time Waveform & Adaptive VAD Segmentation", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, duration)
    ax1.legend(loc="upper right", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    # Panel 2: STFT Spectrogram & Formants
    ax2 = axes[0, 1]
    pcm = ax2.pcolormesh(times_spec, freqs_spec, S_db, shading="gouraud", cmap="magma", vmin=-60, vmax=0)
    ax2.plot(times_formant, f1_tr, color="#fbbf24", linewidth=1.8, label="F1 Track", alpha=0.9)
    ax2.plot(times_formant, f2_tr, color="#38bdf8", linewidth=1.8, label="F2 Track", alpha=0.9)
    ax2.plot(times_formant, f3_tr, color="#f43f5e", linewidth=1.8, label="F3 Track", alpha=0.9)
    ax2.set_title("2. STFT Spectrogram & Formant Resonances (F1, F2, F3)", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, 4000)
    ax2.set_xlim(0, duration)
    fig.colorbar(pcm, ax=ax2, label="Magnitude (dB)")
    ax2.legend(loc="upper right", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    # Panel 3: Pitch Contour
    ax3 = axes[1, 0]
    ax3.plot(times_pitch, pitches, color="#f43f5e", linewidth=1.8, label="Pitch (f0)")
    med_f0 = pitch_stats["median_f0"]
    std_f0 = pitch_stats["std_f0"]
    if med_f0 > 0:
        ax3.axhline(med_f0, color="#38bdf8", linestyle="--", linewidth=1.2, label=f"Median: {med_f0:.1f} Hz")
        ax3.axhspan(max(0, med_f0 - std_f0), med_f0 + std_f0, color="#0284c7", alpha=0.15, label=f"±1σ Range: {pitch_stats['min_f0']:.0f}-{pitch_stats['max_f0']:.0f} Hz")
    ax3.set_title("3. Glottal Fundamental Frequency (f0) Contour", fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Pitch (Hz)")
    ax3.set_ylim(50, 350)
    ax3.set_xlim(0, duration)
    ax3.legend(loc="upper right", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    # Panel 4: Active Vowel LPC Envelope
    ax4 = axes[1, 1]
    windowed = vowel_frame * np.hanning(len(vowel_frame))
    freqs_dft = np.fft.rfftfreq(1024, 1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(windowed, 1024))
    fft_db = 20.0 * np.log10(np.maximum(fft_mag, 1e-5))
    w, h_lpc = np.fft.rfftfreq(1024, 1.0 / fs), 1.0 / np.fft.rfft(lpc_a, 1024)
    lpc_db = 20.0 * np.log10(np.maximum(np.abs(h_lpc), 1e-5))
    lpc_db += np.max(fft_db) - np.max(lpc_db)
    ax4.plot(freqs_dft, fft_db, color="#71717a", alpha=0.5, label="FFT Harmonics")
    ax4.plot(w, lpc_db, color="#38bdf8", linewidth=2.0, label=f"LPC Envelope (p={opt_order})")
    colors = ["#ef4444", "#fbbf24", "#a855f7"]
    for i, f in enumerate(formants[:3]):
        ax4.axvline(f, color=colors[i % len(colors)], linestyle="--", linewidth=1.5, label=f"F{i+1}: {f:.0f} Hz")
    ax4.set_title(f"4. Active Vowel LPC Vocal Tract Filter (t={target_time:.2f}s)", fontweight="bold")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.set_xlim(0, 4000)
    ax4.legend(loc="upper right", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    # Panel 5: MFCCs
    ax5 = axes[2, 0]
    img_mfcc = librosa.display.specshow(mfccs_norm[1:], x_axis="time", sr=fs, hop_length=160, cmap="coolwarm", ax=ax5, vmin=-2.5, vmax=2.5)
    ax5.set_title("5. Normalized Perceptual Timbre: MFCCs (c1 - c12)", fontweight="bold")
    ax5.set_ylabel("MFCC Index")
    ax5.set_yticks(range(0, 12))
    ax5.set_yticklabels([f"c{i}" for i in range(1, 13)])
    fig.colorbar(img_mfcc, ax=ax5, label="Z-Score")

    # Panel 6: Vowel Quadrant
    ax6 = axes[2, 1]
    valid_mask = (~np.isnan(f1_tr)) & (~np.isnan(f2_tr)) & (f1_tr > 200) & (f2_tr > 500)
    if np.any(valid_mask):
        ax6.scatter(f2_tr[valid_mask], f1_tr[valid_mask], color="#38bdf8", alpha=0.35, s=20, label="Voiced Formants")
    ipa_vowels = {"/i/": (2300, 280), "/e/": (2000, 450), "/æ/": (1750, 750), "/ɑ/": (1150, 750), "/u/": (850, 300), "/ʌ/": (1250, 650)}
    for symbol, (f2_ref, f1_ref) in ipa_vowels.items():
        ax6.scatter(f2_ref, f1_ref, color="#f43f5e", marker="x", s=40)
        ax6.text(f2_ref + 20, f1_ref + 10, symbol, color="#fda4af", fontsize=8, fontweight="bold")
    if "f1_centroid" in vowel_space and vowel_space["f1_centroid"] > 0:
        ax6.scatter(vowel_space["f2_centroid"], vowel_space["f1_centroid"], color="#10b981", edgecolors="#ffffff", s=120, marker="*", label=f"Centroid ({vowel_space['dominant_vowel']})", zorder=5)
    ax6.set_xlim(2600, 700)
    ax6.set_ylim(900, 200)
    ax6.set_title(f"6. Phonetic Vowel Space (VSA: {vowel_space.get('vowel_space_area_hz2', 0):,.0f} Hz²)", fontweight="bold")
    ax6.set_xlabel("F2 Frequency (Hz) ← Back vs Front →")
    ax6.set_ylabel("F1 Frequency (Hz) ← Open vs Close →")
    ax6.legend(loc="upper left", facecolor="#18181b", edgecolor="#27272a", labelcolor="#f4f4f5", fontsize=8)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


# -------------------------------------------------------------
# 9. EXPORT ALL ANALYSES BUNDLE (ZIP)
# -------------------------------------------------------------
def export_all_analyses_bundle(audio_file, zip_output_path):
    """
    Generates all 7 individual plots, the 6-panel composite, DFT CSV data points,
    JSON profile, and text report, bundling everything into a downloadable ZIP archive.
    """
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    temp_dir = os.path.join(os.path.dirname(zip_output_path), f"bundle_{base_name}")
    os.makedirs(temp_dir, exist_ok=True)

    files_to_pack = []

    # 1. Individual Plots
    dft_png = os.path.join(temp_dir, "01_dft_magnitude_spectrum.png")
    dft_csv = os.path.join(temp_dir, "01_dft_spectrum_data.csv")
    render_dft_plot(audio_file, save_path=dft_png, csv_save_path=dft_csv)
    files_to_pack.extend([dft_png, dft_csv])

    spec_png = os.path.join(temp_dir, "02_stft_spectrogram.png")
    render_spectrogram_plot(audio_file, save_path=spec_png)
    files_to_pack.append(spec_png)

    pitch_png = os.path.join(temp_dir, "03_pitch_contour_f0.png")
    render_pitch_plot(audio_file, save_path=pitch_png)
    files_to_pack.append(pitch_png)

    lpc_png = os.path.join(temp_dir, "04_lpc_formant_envelope.png")
    render_lpc_plot(audio_file, save_path=lpc_png)
    files_to_pack.append(lpc_png)

    mfcc_png = os.path.join(temp_dir, "05_mfcc_cepstral_cmvn.png")
    render_mfcc_plot(audio_file, save_path=mfcc_png)
    files_to_pack.append(mfcc_png)

    vsa_png = os.path.join(temp_dir, "06_vowel_space_quadrilateral.png")
    render_vowel_space_plot(audio_file, save_path=vsa_png)
    files_to_pack.append(vsa_png)

    glottal_png = os.path.join(temp_dir, "07_glottal_flow_residual.png")
    render_glottal_plot(audio_file, save_path=glottal_png)
    files_to_pack.append(glottal_png)

    dash_png = os.path.join(temp_dir, "00_unified_6panel_dashboard.png")
    render_voice_dashboard(audio_file, save_path=dash_png, show=False)
    files_to_pack.append(dash_png)

    # 2. Text Report & JSON Profile
    from analyze_voice import analyze_audio_signal
    from audio_dsp.diagnostic_engine import generate_voice_diagnostic_report
    import json

    profile, flat = analyze_audio_signal(audio_file)
    rep_text = generate_voice_diagnostic_report(profile)

    rep_path = os.path.join(temp_dir, "clinical_diagnostic_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep_text)
    files_to_pack.append(rep_path)

    json_path = os.path.join(temp_dir, f"profile_{base_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dict(profile), f, indent=2)
    files_to_pack.append(json_path)

    # 3. Create ZIP Archive
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in files_to_pack:
            zipf.write(f, arcname=os.path.basename(f))

    # Clean up temporary individual files
    for f in files_to_pack:
        try:
            os.remove(f)
        except Exception:
            pass
    try:
        os.rmdir(temp_dir)
    except Exception:
        pass

    return zip_output_path


def main():
    parser = argparse.ArgumentParser(description="vox-dsp Diagnostic Multi-Plot & Bundle Generator")
    parser.add_argument("--file", type=str, default="voice.wav", help="Input .wav file")
    parser.add_argument("--all", action="store_true", help="Generate all individual plots and export ZIP bundle")
    parser.add_argument("--dft", action="store_true", help="Generate standalone DFT magnitude spectrum")
    args = parser.parse_args()

    if args.all:
        zip_path = "all_voice_analyses.zip"
        export_all_analyses_bundle(args.file, zip_path)
        print(f"All diagnostic analyses & raw data successfully exported to: {zip_path}")
    elif args.dft:
        render_dft_plot(args.file, "dft_spectrum.png", "dft_spectrum_data.csv")
        print("DFT spectrum plot saved to dft_spectrum.png and data to dft_spectrum_data.csv")
    else:
        render_voice_dashboard(args.file, "voice_dashboard.png")


if __name__ == "__main__":
    main()
