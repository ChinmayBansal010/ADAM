import simpleaudio as sa
import numpy as np
import time

try:
    # Generate a simple sine wave at 44100 Hz (matching your resampled audio)
    sample_rate = 44100
    frequency = 440      # A standard test tone
    duration = 2.0       # seconds

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)

    # Normalize to 16-bit range and convert to int16
    audio_data *= 32767 / np.max(np.abs(audio_data))
    audio_data = audio_data.astype(np.int16)

    print(f"Attempting to play audio: sample_rate={sample_rate}, channels=1, bytes_per_sample=2")
    play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)
    print("Playback initiated. Waiting for it to finish...")
    play_obj.wait_done()
    print("Audio playback successful in test script.")

except sa.SimpleaudioError as e:
    print(f"SimpleaudioError caught during test playback: {e}")
    print("This indicates an issue with simpleaudio or PortAudio.")
except Exception as e:
    print(f"An unexpected error occurred during test playback: {e}")

print("Test script finished.")