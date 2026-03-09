from pydub import AudioSegment
import wave
import struct

# Create a valid dumb wav file
with wave.open('dummy.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(44100)
    data = struct.pack('<h', 0) * 44100
    f.writeframesraw(data)

try:
    print("from_file w/ format:")
    seg = AudioSegment.from_file('dummy.wav', format='wav')
    print("Success with format='wav'")
except Exception as e:
    print("Error format='wav':", e)

try:
    print("from_wav:")
    seg = AudioSegment.from_wav('dummy.wav')
    print("Success from_wav")
except Exception as e:
    print("Error from_wav:", e)

try:
    print("from_file no format:")
    seg = AudioSegment.from_file('dummy.wav')
    print("Success no format")
except Exception as e:
    print("Error no format:", e)
