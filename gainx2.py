import os
import numpy as np
from scipy.io import wavfile

# -------------------------------------------------------
# Parameters
# -------------------------------------------------------
INPUT_FOLDER = r"C:\Users\eyqp2\Desktop\The Collector Pedal - Testing\audio_samples"
OUTPUT_FOLDER = r"C:\Users\eyqp2\Desktop\The Collector Pedal - Testing\processed_audio_samples"
CLIP_GAIN = 8.0  # Gain factor before clipping (higher = more distortion)

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------------------------------
# Distortion function
# -------------------------------------------------------
import numpy as np

def apply_soft_clipping(signal, gain=1.0):
    """
    Apply soft clipping distortion to a float32 signal (-1..1).
    
    Parameters:
        signal : np.ndarray
            Input audio signal normalized to -1..1
        gain : float
            Pre-gain applied before soft clipping
    Returns:
        np.ndarray : Soft-clipped signal
    """
    x = signal * gain
    y = np.zeros_like(x)

    # Apply soft clipping piecewise
    mask1 = np.abs(x) <= 1/3
    mask2 = np.abs(x) > 1/3

    y[mask1] = 2 * x[mask1]  # Gentle linear gain for small values
    y[mask2] = np.sign(x[mask2]) * (1 - (1 - np.abs(x[mask2]))**2)

    # Clip to -1..1 just in case
    y = np.clip(y, -1.0, 1.0)
    return y


# -------------------------------------------------------
# Process all WAVs in folder
# -------------------------------------------------------
for f in os.listdir(INPUT_FOLDER):
    if not f.lower().endswith(".wav"):
        continue

    input_path = os.path.join(INPUT_FOLDER, f)
    sr, audio_raw = wavfile.read(input_path)

    # Normalize to -1..1 if needed
    if audio_raw.dtype == np.int16:
        audio = audio_raw.astype(np.float32) / 32768.0
    else:
        audio = audio_raw.astype(np.float32)

    # Apply distortion to create target
    target = apply_soft_clipping(audio, gain=CLIP_GAIN)

    # Prepare filenames
    base_name = os.path.splitext(f)[0]
    #input_out_path = os.path.join(OUTPUT_FOLDER, base_name + "_input.wav")
    target_out_path = os.path.join(OUTPUT_FOLDER, base_name + "_target.wav")

    # Convert back to int16
    input_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    target_int16 = np.clip(target * 32767, -32768, 32767).astype(np.int16)

    # Save files
    #wavfile.write(input_out_path, sr, input_int16)
    wavfile.write(target_out_path, sr, target_int16)

    print(f"Processed {f} target -> {target_out_path}")

print("All files processed.")
