"""
Unified 6-Panel Diagnostic Speech & Biometric Dashboard.
Renders publication-grade acoustic visualizations combining:
1. Waveform with Adaptive VAD Segmentation
2. STFT Spectrogram with Overlaid Formant Tracks (F1, F2, F3)
3. Glottal Pitch (f0) Intonation Contour
4. Active Vowel Frame LPC Vocal Tract Envelope vs FFT Harmonics
5. CMVN Normalized MFCCs (c1-c12) and Log Energy (c0)
6. Phonetic Vowel Quadrilateral (F1 vs F2 IPA Acoustic Space)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def render_voice_dashboard(audio_file="voice.wav", save_path="voice_dashboard.png", show=True):
    """
    Renders and exports the unified 6-panel voice diagnostic dashboard.
    """
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    signal, fs = librosa.load(audio_file, sr=16000, mono=True)
    duration = len(signal) / fs
    times_wav = np.linspace(0, duration, len(signal))

    # Run DSP Extractions
    times_vad, vad_labels, energies_db, _ = compute_vad_segments(signal, fs)
    vowel_start, vowel_end, _ = extract_longest_voiced_segment(signal, fs)
    times_pitch, pitches, _ = track_pitch_contour(signal, fs)
    pitch_stats = compute_pitch_statistics(pitches)

    # LPC Formants on the clean vowel segment
    target_time = (vowel_start + vowel_end) / 2.0
    frame_len = int(0.03 * fs)
    start_idx = max(0, min(len(signal) - frame_len, int(target_time * fs - frame_len / 2)))
    vowel_frame = signal[start_idx : start_idx + frame_len]

    prelim_formants, _, _ = extract_lpc_formants(vowel_frame, fs, lpc_order=18)
    f1_init = prelim_formants[0] if len(prelim_formants) > 0 else 500.0
    f2_init = prelim_formants[1] if len(prelim_formants) > 1 else 1500.0
    f3_init = prelim_formants[2] if len(prelim_formants) > 2 else 2500.0
    vtl_cm, demographic, opt_order = estimate_vocal_tract_length(
        pitch_stats["median_f0"], f1_init, f2_init, f3_init
    )

    formants, bandwidths, lpc_a = extract_lpc_formants(
        vowel_frame, fs, lpc_order=opt_order, pre_emphasis=0.97, max_bandwidth=400.0
    )

    # Formant tracks across time
    times_formant, f1_tr, f2_tr, f3_tr = track_formants_contour(
        signal, fs, vad_labels=vad_labels, lpc_order=opt_order
    )
    vowel_space = compute_vowel_space(f1_tr, f2_tr)

    # STFT Spectrogram
    times_spec, freqs_spec, S_db = compute_stft_spectrogram(signal, fs, fmax=4000.0)

    # MFCCs with CMVN
    mfccs_raw, mfccs_norm, _, _, _, times_mfcc = extract_mfcc_features(
        signal, fs, vad_labels=vad_labels
    )

    # Initialize Figure Layout (3 rows x 2 cols)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        f"vox-dsp Unified Voice Diagnostic Dashboard: {os.path.basename(audio_file)} "
        f"({demographic}, Est. VTL: {vtl_cm} cm)",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    # -------------------------------------------------------------
    # PANEL 1: Waveform & Adaptive VAD
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(times_wav, signal, color="#2b5c8f", alpha=0.7, linewidth=0.7, label="Waveform")

    # Overlay colored VAD regions
    dt = times_vad[1] - times_vad[0] if len(times_vad) > 1 else 0.01
    for t, label in zip(times_vad, vad_labels):
        if label == "voiced":
            ax1.axvspan(t - dt / 2, t + dt / 2, color="forestgreen", alpha=0.15)
        elif label == "unvoiced":
            ax1.axvspan(t - dt / 2, t + dt / 2, color="orange", alpha=0.15)

    # Highlight extracted clean vowel segment
    ax1.axvspan(
        vowel_start,
        vowel_end,
        color="crimson",
        alpha=0.25,
        label=f"Clinical Vowel ({vowel_start:.2f}s - {vowel_end:.2f}s)",
    )

    ax1.set_title("1. Time Waveform & Adaptive VAD Segmentation", fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, duration)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # PANEL 2: STFT Spectrogram & Overlaid Formants
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    pcm = ax2.pcolormesh(
        times_spec,
        freqs_spec,
        S_db,
        shading="gouraud",
        cmap="magma",
        vmin=-60,
        vmax=0,
    )
    # Overlay dynamic formant curves
    ax2.plot(times_formant, f1_tr, color="gold", linewidth=1.8, label="F1 Track", alpha=0.9)
    ax2.plot(times_formant, f2_tr, color="cyan", linewidth=1.8, label="F2 Track", alpha=0.9)
    ax2.plot(times_formant, f3_tr, color="#ff4081", linewidth=1.8, label="F3 Track", alpha=0.9)

    ax2.set_title("2. STFT Spectrogram & Formant Resonances (F1, F2, F3)", fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_ylim(0, 4000)
    ax2.set_xlim(0, duration)
    fig.colorbar(pcm, ax=ax2, label="Magnitude (dB)")
    ax2.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # PANEL 3: Glottal Pitch (f0) Intonation Contour
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(times_pitch, pitches, color="crimson", linewidth=1.8, label="Pitch (f0)")

    # Baseline median and +/- 1 stddev band
    med_f0 = pitch_stats["median_f0"]
    std_f0 = pitch_stats["std_f0"]
    if med_f0 > 0:
        ax3.axhline(med_f0, color="navy", linestyle="--", linewidth=1.2, label=f"Median: {med_f0:.1f} Hz")
        ax3.axhspan(
            max(0, med_f0 - std_f0),
            med_f0 + std_f0,
            color="navy",
            alpha=0.1,
            label=f"±1σ Range: {pitch_stats['min_f0']:.0f}-{pitch_stats['max_f0']:.0f} Hz",
        )

    ax3.set_title("3. Glottal Fundamental Frequency ($f_0$) Contour", fontweight="bold")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Pitch (Hz)")
    ax3.set_ylim(50, 350)
    ax3.set_xlim(0, duration)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # PANEL 4: Active Vowel Frame LPC Envelope vs FFT Harmonics
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    # FFT spectrum of vowel frame
    windowed = vowel_frame * np.hanning(len(vowel_frame))
    freqs_dft = np.fft.rfftfreq(1024, 1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(windowed, 1024))
    fft_db = 20.0 * np.log10(np.maximum(fft_mag, 1e-5))

    # LPC filter frequency response
    w, h_lpc = np.fft.rfftfreq(1024, 1.0 / fs), 1.0 / np.fft.rfft(lpc_a, 1024)
    lpc_db = 20.0 * np.log10(np.maximum(np.abs(h_lpc), 1e-5))
    lpc_db += np.max(fft_db) - np.max(lpc_db)  # Offset alignment

    ax4.plot(freqs_dft, fft_db, color="gray", alpha=0.5, label="FFT Harmonics")
    ax4.plot(w, lpc_db, color="navy", linewidth=2.0, label=f"LPC Envelope (order={opt_order})")

    # Mark detected formants
    colors = ["crimson", "darkorange", "purple"]
    for i, f in enumerate(formants[:3]):
        c = colors[i % len(colors)]
        ax4.axvline(f, color=c, linestyle="--", linewidth=1.5, label=f"F{i+1}: {f:.0f} Hz")

    ax4.set_title(f"4. Active Vowel LPC Vocal Tract Filter (t={target_time:.2f}s)", fontweight="bold")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Magnitude (dB)")
    ax4.set_xlim(0, 4000)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # PANEL 5: CMVN Normalized MFCCs & Energy Contour
    # -------------------------------------------------------------
    ax5 = axes[2, 0]
    img_mfcc = librosa.display.specshow(
        mfccs_norm[1:],
        x_axis="time",
        sr=fs,
        hop_length=160,
        cmap="coolwarm",
        ax=ax5,
        vmin=-2.5,
        vmax=2.5,
    )
    ax5.set_title("5. Normalized Perceptual Timbre: MFCCs ($c_1 - c_{12}$)", fontweight="bold")
    ax5.set_ylabel("MFCC Index")
    ax5.set_yticks(range(0, 12))
    ax5.set_yticklabels([f"c{i}" for i in range(1, 13)])
    fig.colorbar(img_mfcc, ax=ax5, label="Z-Score")

    # -------------------------------------------------------------
    # PANEL 6: Phonetic Vowel Quadrilateral (F1 vs F2 IPA Space)
    # -------------------------------------------------------------
    ax6 = axes[2, 1]
    # Filter valid F1 and F2 points from contour
    valid_mask = (~np.isnan(f1_tr)) & (~np.isnan(f2_tr)) & (f1_tr > 200) & (f2_tr > 500)
    if np.any(valid_mask):
        ax6.scatter(
            f2_tr[valid_mask],
            f1_tr[valid_mask],
            color="#3f51b5",
            alpha=0.4,
            s=20,
            label="Voiced Frame Formants",
        )

    # Reference IPA cardinal vowel regions (approximate adult formant norms)
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
        ax6.scatter(f2_ref, f1_ref, color="crimson", marker="x", s=50)
        ax6.text(f2_ref + 20, f1_ref + 10, symbol, color="darkred", fontsize=8, fontweight="bold")

    # Highlight speaker centroid
    if "f1_centroid" in vowel_space and vowel_space["f1_centroid"] > 0:
        ax6.scatter(
            vowel_space["f2_centroid"],
            vowel_space["f1_centroid"],
            color="lime",
            edgecolors="black",
            s=120,
            marker="*",
            label=f"Speaker Centroid ({vowel_space['dominant_vowel']})",
            zorder=5,
        )

    # Inverted axes per acoustic phonetics standard (top-left is /i/, bottom-right is /ɑ/)
    ax6.set_xlim(2600, 700)
    ax6.set_ylim(900, 200)
    ax6.set_title(
        f"6. Phonetic Vowel Space (VSA: {vowel_space['vowel_space_area_hz2']:,.0f} Hz²)",
        fontweight="bold",
    )
    ax6.set_xlabel("F2 Frequency (Hz) ← Back vs Front →")
    ax6.set_ylabel("F1 Frequency (Hz) ← Open vs Close →")
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="upper left", fontsize=8)

    # Save and show
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Unified Diagnostic Dashboard successfully saved to: {save_path}")

    if show and "FigureCanvasAgg" not in type(fig.canvas).__name__:
        try:
            plt.show()
        except Exception:
            pass

    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Generate vox-dsp 6-panel diagnostic dashboard")
    parser.add_argument("--file", type=str, default="voice.wav", help="Input .wav file")
    parser.add_argument("--output", type=str, default="voice_dashboard.png", help="Output PNG file path")
    args = parser.parse_args()
    render_voice_dashboard(args.file, args.output)


if __name__ == "__main__":
    main()

