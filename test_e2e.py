import os
import io
import wave
import struct
from app import clone_voice

# Create a valid dumb wav file
with wave.open('dummy.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    data = struct.pack('<h', 0) * 44100
    f.writeframesraw(data)

class DummyProgress:
    def __call__(self, value, desc=""):
        print(f"Progress: {value} - {desc}")

print("Testing clone_voice...")
res = clone_voice(
    ref_audio="dummy.wav",
    ref_text="Test",
    gen_text="Xin chào",
    speed=1.0,
    progress=DummyProgress()
)

print("Result:", res)
