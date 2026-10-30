import parselmouth
from parselmouth.praat import call

sound = parselmouth.Sound("voice.wav")

# Extract the stable voiced segment (2.5s to 3.5s)
voiced_segment = sound.extract_part(from_time=2.5, to_time=3.5, preserve_times=True)

# Generate pitch and glottal pulse tracking for the clean vowel
pitch = voiced_segment.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=300.0)
point_process = call(voiced_segment, "To PointProcess (periodic, cc)", 75.0, 300.0)

# Re-calculate perturbation metrics on the vowel
jitter_local = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100
shimmer_local = call([voiced_segment, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6) * 100
harmonicity = voiced_segment.to_harmonicity(time_step=0.01, minimum_pitch=75.0)
hnr_mean = call(harmonicity, "Get mean", 0, 0)

print("=== Clean Voiced Vowel Segment (2.5s - 3.5s) ===")
print(f"Jitter (local)  : {jitter_local:.2f}%     (Normal: < 1.0%)")
print(f"Shimmer (local) : {shimmer_local:.2f}%     (Normal: < 3.81%)")
print(f"Mean HNR        : {hnr_mean:.2f} dB   (Normal: > 15-20 dB)")