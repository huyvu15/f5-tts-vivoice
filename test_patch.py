import pydub

orig_from_file = pydub.AudioSegment.from_file

def patched_from_file(file, format=None, **kwargs):
    if format is None and str(file).lower().endswith(".wav"):
        format = "wav"
    return orig_from_file(file, format=format, **kwargs)

pydub.AudioSegment.from_file = patched_from_file

print("Patched from_file")
