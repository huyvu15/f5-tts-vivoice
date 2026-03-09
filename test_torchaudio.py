import torchaudio
import soundfile as sf
import torch

def patched_torchaudio_load(filepath, **kwargs):
    data, sr = sf.read(filepath)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data = data.T
    return torch.from_numpy(data.copy()).float(), sr

torchaudio.load = patched_torchaudio_load

import wave
import struct

with wave.open('dummy.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    data = struct.pack('<h', 0) * 44100
    f.writeframesraw(data)

try:
    audio, sr = torchaudio.load('dummy.wav')
    print("Success:", audio.shape, sr)
except Exception as e:
    import traceback
    traceback.print_exc()
