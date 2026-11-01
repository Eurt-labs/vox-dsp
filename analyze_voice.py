"""
Master Voice Analysis Pipeline.
Combines all 5 DSP layers, adaptive VAD, demographic morphology,
clinical perturbation biometrics, and automated diagnostic report generation.

Usage:
  python analyze_voice.py                          # Analyze default voice.wav
  python analyze_voice.py --file audio.wav         # Analyze specific file
  python analyze_voice.py --batch ./samples/       # Batch process folder into CSV
  python analyze_voice.py --json                   # Output machine-readable JSON
  python analyze_voice.py --dashboard              # Generate 6-panel visual dashboard
"""

import os
import sys
import glob
import json
import csv
import argparse
import numpy as np
import soundfile as sf
import librosa

from audio_dsp.vad import compute_vad_segments, extract_longest_voiced_segment
from audio_dsp.pitch import track_pitch_contour, compute_pitch_statistics
from audio_dsp.formants import (
    extract_lpc_formants,
    track_formants_contour,
    estimate_vocal_tract_length,
    compute_vowel_space,
)
from audio_dsp.spectral import compute_active_frame_dft
from audio_dsp.mfcc import extract_mfcc_features
from audio_dsp.biometrics import (
    compute_clinical_voice_quality,
    compute_cpps,
    glottal_inverse_filtering,
)
from audio_dsp.diagnostic_engine import (
    evaluate_normative_thresholds,
    generate_voice_diagnostic_report,
)


class VoiceProfile(dict):
    """
    Dual-interface container:
    - If unpacked (`profile, flat = analyze_audio_signal(path)`): behaves as a 2-tuple.
    - If treated as a dict (`profile = analyze_audio_signal(path)`): behaves as a flat dict for CSV.
    """
    def __init__(self, profile_dict, flat_dict):
        super().__init__(flat_dict)
        self.profile = profile_dict
        self.flat = flat_dict

    def __iter__(self):
        return iter([self.profile, self.flat])

    def __getitem__(self, key):
        if key in self.profile:
            return self.profile[key]
        return super().__getitem__(key)


def load_audio(audio_path, target_sr=16000):
    """Loads audio, normalizes, resamples to target_sr if needed, and downmixes to mono."""
    signal, fs = sf.read(audio_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if target_sr is not None and fs != target_sr:
        signal = librosa.resample(signal, orig_sr=fs, target_sr=target_sr)
        fs = target_sr
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val
    return signal, fs


def save_rows_to_csv(rows, output_path):
    """Saves rows (list of dicts) to CSV using the standard library csv module."""
    if isinstance(rows, dict):
        rows = [rows]
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    existing = []

    if os.path.isfile(output_path):
        try:
            with open(output_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    for fld in reader.fieldnames:
                        if fld not in fieldnames:
                            fieldnames.append(fld)
                for r in reader:
                    existing.append(r)
        except Exception:
            existing = []

    new_files = {r.get("File") for r in rows}
    merged_rows = [r for r in existing if r.get("File") not in new_files] + rows

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in merged_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def analyze_audio_signal(audio_path):
    """
    Executes the full intelligent voice analysis pipeline on a single audio file.

    Returns:
        profile (dict): Hierarchical diagnostic profile.
        flat_features (dict): 1D dictionary suitable for tabular CSV datasets.
    """
    signal, fs = load_audio(audio_path)
    duration_s = len(signal) / fs

    # 1. Adaptive VAD & Voiced Segment Extraction
    times_vad, vad_labels, energies_db, zcr = compute_vad_segments(signal, fs)
    vowel_start, vowel_end, vowel_seg = extract_longest_voiced_segment(signal, fs)

    # 2. Fundamental Pitch (f0) Tracking
    times_pitch, pitches, voicing_strength = track_pitch_contour(signal, fs)
    pitch_stats = compute_pitch_statistics(pitches)

    # 3. Active Vowel Frame & LPC Formants
    target_vowel_time = (vowel_start + vowel_end) / 2.0
    freqs_dft, mag_dft, peak_freq, peak_db, frame_time = compute_active_frame_dft(
        signal, fs, target_time=target_vowel_time
    )

    frame_len = int(0.03 * fs)
    start_idx = max(0, min(len(signal) - frame_len, int(target_vowel_time * fs - frame_len / 2)))
    vowel_frame = signal[start_idx : start_idx + frame_len]

    prelim_formants, _, _ = extract_lpc_formants(vowel_frame, fs, lpc_order=18)
    f1_init = prelim_formants[0] if len(prelim_formants) > 0 else 500.0
    f2_init = prelim_formants[1] if len(prelim_formants) > 1 else 1500.0
    f3_init = prelim_formants[2] if len(prelim_formants) > 2 else 2500.0

    vtl_cm, demographic_class, optimal_order = estimate_vocal_tract_length(
        pitch_stats["median_f0"], f1_init, f2_init, f3_init
    )

    formants, bandwidths, lpc_a = extract_lpc_formants(
        vowel_frame, fs, lpc_order=optimal_order, pre_emphasis=0.97, max_bandwidth=400.0
    )

    _, f1_tr, f2_tr, f3_tr = track_formants_contour(
        signal, fs, vad_labels=vad_labels, lpc_order=optimal_order
    )
    vowel_space = compute_vowel_space(f1_tr, f2_tr)

    # 4. Perceptual Timbre (MFCCs + CMVN + Deltas)
    _, _, _, _, mfcc_summary, _ = extract_mfcc_features(
        signal, fs, vad_labels=vad_labels
    )

    # 5. Clinical Voice Quality (Praat perturbation on auto-detected vowel)
    clinical_vowel = compute_clinical_voice_quality(
        audio_path, start_time=vowel_start, end_time=vowel_end
    )

    # 6. Continuous Speech Dysphonia (CPPS)
    cpps_mean, cpps_std = compute_cpps(signal, fs)

    # 7. Glottal Inverse Filtering (Source residual e[n])
    residual, naq = glottal_inverse_filtering(signal, fs, lpc_coeffs=lpc_a)

    # 8. Automated Clinical & Acoustic Diagnosis
    combined_metrics = {
        **pitch_stats,
        **clinical_vowel,
        "cpps_db": cpps_mean,
        "naq": naq,
    }
    diagnostic_eval = evaluate_normative_thresholds(combined_metrics)

    profile = {
        "file": os.path.basename(audio_path),
        "duration_s": round(duration_s, 2),
        "sampling_rate": fs,
        "demographics": {
            "demographic_class": demographic_class,
            "vocal_tract_length_cm": vtl_cm,
            "adapted_lpc_order": optimal_order,
        },
        "pitch": pitch_stats,
        "spectral_peak": {
            "dominant_frequency_hz": peak_freq,
            "level_db": peak_db,
            "active_frame_time_s": frame_time,
        },
        "formants": {
            "f1_hz": formants[0] if len(formants) > 0 else 0.0,
            "f2_hz": formants[1] if len(formants) > 1 else 0.0,
            "f3_hz": formants[2] if len(formants) > 2 else 0.0,
            "dominant_vowel": vowel_space["dominant_vowel"],
            "vowel_space_area_hz2": vowel_space["vowel_space_area_hz2"],
        },
        "clinical_vowel_biometrics": clinical_vowel,
        "continuous_speech_biometrics": {
            "cpps_db": cpps_mean,
            "cpps_std_db": cpps_std,
            "normalized_amplitude_quotient": naq,
        },
        "diagnostic_evaluation": diagnostic_eval,
    }

    flat = {
        "File": os.path.basename(audio_path),
        "Duration_s": round(duration_s, 2),
        "Sample_Rate_Hz": fs,
        "Demographic": demographic_class,
        "Est_VTL_cm": vtl_cm,
        "Pitch_Median_Hz": pitch_stats["median_f0"],
        "Pitch_Mean_Hz": pitch_stats["mean_f0"],
        "Pitch_StdDev_Hz": pitch_stats["std_f0"],
        "Pitch_Range_Semitones": pitch_stats["pitch_range_semitones"],
        "Voiced_Ratio": pitch_stats["voiced_ratio"],
        "Formant_F1_Hz": formants[0] if len(formants) > 0 else 0.0,
        "Formant_F2_Hz": formants[1] if len(formants) > 1 else 0.0,
        "Formant_F3_Hz": formants[2] if len(formants) > 2 else 0.0,
        "Dominant_Vowel": vowel_space["dominant_vowel"],
        "Vowel_Space_Area_Hz2": vowel_space["vowel_space_area_hz2"],
        "Jitter_Local_Pct": clinical_vowel["jitter_local_percent"],
        "Jitter_RAP_Pct": clinical_vowel["jitter_rap_percent"],
        "Shimmer_Local_Pct": clinical_vowel["shimmer_local_percent"],
        "Shimmer_APQ5_Pct": clinical_vowel["shimmer_apq5_percent"],
        "HNR_dB": clinical_vowel["hnr_db"],
        "CPPS_dB": cpps_mean,
        "NAQ": naq,
        "Clinical_Assessment": diagnostic_eval["overall_health"],
    }
    for k, v in mfcc_summary.items():
        flat[k] = v

    return VoiceProfile(profile, flat)


def main():
    parser = argparse.ArgumentParser(description="vox-dsp: Intelligent Voice Signal Processing & Biometric Pipeline")
    parser.add_argument("--file", type=str, default="voice.wav", help="Path to single .wav file to analyze")
    parser.add_argument("--batch", type=str, default=None, help="Directory of .wav files to batch process")
    parser.add_argument("--output", type=str, default="voice_dataset.csv", help="CSV export destination")
    parser.add_argument("--json", action="store_true", help="Print structured JSON profile to stdout")
    parser.add_argument("--dashboard", action="store_true", help="Launch 6-panel diagnostic dashboard")
    parser.add_argument("--web", action="store_true", help="Launch centralized access dashboard web application")
    args = parser.parse_args()

    if args.web:
        try:
            from app import start_server
            start_server(open_browser=True)
        except Exception as e:
            print(f"Error starting dashboard server: {e}")
        return

    if args.batch:
        wav_files = glob.glob(os.path.join(args.batch, "*.wav"))
        if not wav_files:
            print(f"No .wav files found in directory: {args.batch}")
            sys.exit(1)

        print(f"Batch processing {len(wav_files)} files in '{args.batch}'...")
        rows = []
        for f in wav_files:
            try:
                _, flat = analyze_audio_signal(f)
                rows.append(flat)
                print(f"  Processed: {os.path.basename(f)}")
            except Exception as e:
                print(f"  Error processing {os.path.basename(f)}: {e}")

        save_rows_to_csv(rows, args.output)
        print(f"\nSuccessfully exported {len(rows)} audio profiles to: {args.output}")

    else:
        if not os.path.isfile(args.file):
            print(f"Audio file not found: {args.file}")
            sys.exit(1)

        profile, flat = analyze_audio_signal(args.file)

        if args.json:
            print(json.dumps(profile, indent=2))
        else:
            report_text = generate_voice_diagnostic_report(profile)
            print(report_text)

        save_rows_to_csv([flat], args.output)
        print(f"\nFeature profile saved to: {args.output}")

        if args.dashboard:
            try:
                from dashboard import render_voice_dashboard
                render_voice_dashboard(args.file)
            except ImportError as e:
                print(f"Could not load dashboard: {e}")


if __name__ == "__main__":
    main()
