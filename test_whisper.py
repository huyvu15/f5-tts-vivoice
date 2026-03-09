from transformers import pipeline

print("Loading pipeline...")
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo")

print("Transcribing dummy.wav...")
try:
    res = asr_pipe('dummy.wav')
    print("Success:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
