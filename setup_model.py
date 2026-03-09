import os
import shutil
from huggingface_hub import snapshot_download

def prepare_model():
    print("Downloading model...")
    model_dir = snapshot_download(
        repo_id="hynt/F5-TTS-Vietnamese-ViVoice",
        local_dir="./F5-TTS-Vietnamese-ViVoice"
    )
    print(f"Model downloaded to: {model_dir}")
    
    src = "./F5-TTS-Vietnamese-ViVoice/config.json"
    dst = "./F5-TTS-Vietnamese-ViVoice/vocab.txt"
    if os.path.exists(src):
        shutil.copy(src, dst)
        print("✅ Copied config.json -> vocab.txt")
    
    os.makedirs("./tests/outputs", exist_ok=True)
    os.makedirs("./outputs", exist_ok=True)
    print("✅ Created output directories")
    
    assert os.path.exists("./F5-TTS-Vietnamese-ViVoice/vocab.txt")
    assert os.path.exists("./F5-TTS-Vietnamese-ViVoice/model_last.pt")
    print("✅ Requirements met! You can now run app.py")

if __name__ == "__main__":
    prepare_model()
