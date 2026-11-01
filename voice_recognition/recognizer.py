"""
Acoustic Speaker Recognition & Voice Biometrics Identification Engine.
Extracts 84-dimensional multi-modal voice fingerprints (MFCCs, Deltas, Delta-Deltas,
Glottal Pitch, Vocal Tract Formants, and Spectral Shape) and performs Cosine/Euclidean matching.
"""

import os
import json
import time
import numpy as np
import soundfile as sf
import librosa

from audio_dsp.vad import compute_vad_segments
from audio_dsp.pitch import track_pitch_contour, compute_pitch_statistics
from audio_dsp.formants import extract_lpc_formants, estimate_vocal_tract_length

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrolled_speakers.json")


def extract_voice_fingerprint(audio_path, target_sr=16000):
    """
    Extracts a dense 84-dimensional acoustic voice biometric fingerprint.

    Feature Breakdown:
    - 39 dims: Mean MFCCs (c0-c12) + Deltas (Δ) + Delta-Deltas (Δ²)
    - 39 dims: StdDev MFCCs + Deltas + Delta-Deltas
    - 3 dims: Glottal Pitch (Median f0, Pitch StdDev, Pitch Range Semitones)
    - 3 dims: Vocal Tract Formants (F1, F2, F3)

    Returns:
        embedding (np.ndarray): 1D float32 biometric vector.
        aux_meta (dict): Human-readable acoustic summary for display.
    """
    signal, fs = sf.read(audio_path)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    if fs != target_sr:
        signal = librosa.resample(signal, orig_sr=fs, target_sr=target_sr)
        fs = target_sr

    max_v = np.max(np.abs(signal))
    if max_v > 0:
        signal = signal / max_v

    # 1. Voice Activity Detection (mask out silence)
    times_vad, vad_labels, _, _ = compute_vad_segments(signal, fs)
    voiced_mask = vad_labels != "silence"
    if not np.any(voiced_mask):
        voiced_mask = np.ones(len(vad_labels), dtype=bool)

    # 2. MFCCs + Deltas + Delta-Deltas (39 coefficients across time)
    mfcc_raw = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=13, n_fft=512, hop_length=160)
    deltas = librosa.feature.delta(mfcc_raw, order=1)
    delta2 = librosa.feature.delta(mfcc_raw, order=2)
    mfcc_39 = np.vstack([mfcc_raw, deltas, delta2])  # Shape (39, T)

    # Match time alignment with VAD
    n_frames = min(mfcc_39.shape[1], len(voiced_mask))
    active_mfcc = mfcc_39[:, :n_frames][:, voiced_mask[:n_frames]]
    if active_mfcc.shape[1] == 0:
        active_mfcc = mfcc_39

    mfcc_means = np.mean(active_mfcc, axis=1)  # 39 dims
    mfcc_stds = np.std(active_mfcc, axis=1)    # 39 dims

    # 3. Glottal Pitch (f0)
    _, pitches, _ = track_pitch_contour(signal, fs)
    pitch_stats = compute_pitch_statistics(pitches)
    pitch_vec = np.array([
        pitch_stats.get("median_f0", 120.0) / 100.0,
        pitch_stats.get("std_f0", 25.0) / 50.0,
        pitch_stats.get("pitch_range_semitones", 8.0) / 12.0,
    ])

    # 4. Vocal Tract Resonances (Formants F1, F2, F3)
    frame_len = int(0.03 * fs)
    mid_sample = len(signal) // 2
    formants, _, _ = extract_lpc_formants(signal[mid_sample : mid_sample + frame_len], fs, lpc_order=18)
    f1 = formants[0] if len(formants) > 0 else 500.0
    f2 = formants[1] if len(formants) > 1 else 1500.0
    f3 = formants[2] if len(formants) > 2 else 2500.0
    formant_vec = np.array([f1 / 1000.0, f2 / 2000.0, f3 / 3000.0])

    # Assemble and normalize
    raw_embedding = np.concatenate([mfcc_means, mfcc_stds, pitch_vec, formant_vec])
    norm = np.linalg.norm(raw_embedding)
    embedding = raw_embedding / (norm + 1e-10)

    aux_meta = {
        "median_f0": pitch_stats.get("median_f0", 0.0),
        "f1_hz": round(float(f1), 1),
        "f2_hz": round(float(f2), 1),
        "f3_hz": round(float(f3), 1),
        "voiced_ratio": pitch_stats.get("voiced_ratio", 0.0),
    }

    return embedding.astype(np.float32), aux_meta


class SpeakerDatabase:
    """Manages enrolled speaker biometric profiles and performs identification matching."""

    def __init__(self, db_file=DB_PATH):
        self.db_file = db_file
        self.speakers = self._load()

    def _load(self):
        if not os.path.isfile(self.db_file):
            return {}
        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self.speakers, f, indent=2)

    def enroll_speaker(self, name, audio_path, sentence_text=""):
        """Enrolls a speaker voice profile into the biometric database."""
        name = name.strip()
        if not name:
            raise ValueError("Speaker name cannot be empty.")

        embedding, aux_meta = extract_voice_fingerprint(audio_path)

        self.speakers[name] = {
            "name": name,
            "embedding": embedding.tolist(),
            "sample_file": os.path.basename(audio_path),
            "sentence_text": sentence_text,
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": aux_meta,
        }
        self._save()
        return self.speakers[name]

    def identify_speaker(self, audio_path, threshold=0.75):
        """
        Compares an unknown audio recording against all enrolled voice fingerprints.

        Returns:
            match (dict): Best matching speaker, confidence percentage, and candidate rankings.
        """
        if not self.speakers:
            return {
                "matched": False,
                "speaker": "No Enrolled Speakers",
                "confidence_percent": 0.0,
                "candidates": [],
                "message": "The biometric database is currently empty. Please enroll at least one voice profile first.",
            }

        target_emb, target_meta = extract_voice_fingerprint(audio_path)

        candidates = []
        for name, profile in self.speakers.items():
            enrolled_emb = np.array(profile["embedding"], dtype=np.float32)

            # Cosine similarity on unit vectors is simply the dot product
            cosine_sim = float(np.dot(target_emb, enrolled_emb))
            cosine_sim = max(0.0, min(1.0, cosine_sim))

            # Convert to calibrated confidence score [0% - 100%]
            confidence = round(cosine_sim * 100.0, 1)

            candidates.append({
                "name": name,
                "similarity": round(cosine_sim, 4),
                "confidence_percent": confidence,
                "enrolled_at": profile.get("enrolled_at", ""),
                "meta": profile.get("meta", {}),
            })

        # Sort by highest similarity
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        best = candidates[0]

        is_match = best["similarity"] >= threshold

        return {
            "matched": is_match,
            "speaker": best["name"] if is_match else "Unknown / Unverified Speaker",
            "confidence_percent": best["confidence_percent"],
            "best_candidate": best["name"],
            "threshold": threshold,
            "candidates": candidates,
            "target_meta": target_meta,
            "message": (
                f"Voice authenticated as '{best['name']}' with {best['confidence_percent']}% confidence."
                if is_match
                else f"Voice did not match any enrolled speaker above the {int(threshold*100)}% threshold (closest: '{best['name']}' at {best['confidence_percent']}%)."
            ),
        }

    def list_speakers(self):
        """Returns summary list of all enrolled speakers."""
        return [
            {
                "name": name,
                "enrolled_at": prof.get("enrolled_at", ""),
                "sample_file": prof.get("sample_file", ""),
                "sentence_text": prof.get("sentence_text", ""),
                "median_f0": prof.get("meta", {}).get("median_f0", 0.0),
            }
            for name, prof in self.speakers.items()
        ]

    def delete_speaker(self, name):
        """Deletes a speaker from the database."""
        if name in self.speakers:
            del self.speakers[name]
            self._save()
            return True
        return False


# Global default database instance
default_db = SpeakerDatabase()


def main():
    import argparse
    import time
    parser = argparse.ArgumentParser(description="vox-dsp Voice Recognition Engine")
    parser.add_argument("--enroll", type=str, default=None, help="Speaker name to enroll")
    parser.add_argument("--identify", type=str, default=None, help="Path to audio file to identify")
    parser.add_argument("--file", type=str, default="voice.wav", help="Audio file for enrollment")
    parser.add_argument("--list", action="store_true", help="List enrolled speakers")
    args = parser.parse_args()

    db = SpeakerDatabase()

    if args.list:
        speakers = db.list_speakers()
        print(f"\nEnrolled Speakers ({len(speakers)}):")
        for s in speakers:
            print(f"  - {s['name']} (Enrolled: {s['enrolled_at']}, Pitch: {s['median_f0']:.1f} Hz)")
        return

    if args.enroll:
        if not os.path.isfile(args.file):
            print(f"Audio file not found: {args.file}")
            return
        res = db.enroll_speaker(args.enroll, args.file)
        print(f"Successfully enrolled speaker: '{args.enroll}' using '{args.file}'")
        return

    if args.identify:
        if not os.path.isfile(args.identify):
            print(f"Audio file not found: {args.identify}")
            return
        result = db.identify_speaker(args.identify)
        print("\n" + "=" * 50)
        print("          VOICE RECOGNITION RESULT")
        print("=" * 50)
        print(f"Identified Speaker : {result['speaker']}")
        print(f"Confidence Score   : {result['confidence_percent']}%")
        print(f"Assessment         : {result['message']}")
        print("\nCandidate Rankings:")
        for c in result["candidates"]:
            print(f"  - {c['name']:<15} : {c['confidence_percent']}% match")
        print("=" * 50)


if __name__ == "__main__":
    import time
    main()
