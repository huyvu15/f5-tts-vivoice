import torchaudio

import os
from f5_tts.infer.utils_infer import load_vocoder, load_model, infer_process
from f5_tts.model import DiT
from importlib.resources import files
from omegaconf import OmegaConf

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
VOCAB_FILE = os.path.join(BASE_DIR, "F5-TTS-Vietnamese-ViVoice", "vocab.txt")
CKPT_FILE  = os.path.join(BASE_DIR, "F5-TTS-Vietnamese-ViVoice", "model_last.pt")

print("Loading Vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False)

print("Loading model config...")
model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml"))).model
model_cls = globals()[model_cfg.backbone]

print("Loading Model...")
ema_model = load_model(model_cls, model_cfg.arch, CKPT_FILE, mel_spec_type="vocos", vocab_file=VOCAB_FILE)

print("Model loaded successfully!")
