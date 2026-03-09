import gradio as gr
import os
import shutil
import soundfile as sf
import torch
from datetime import datetime
from pydub import AudioSegment, silence

import sys

# --- HỖ TRỢ WINDOWS ---
# Sửa lỗi UnicodeEncodeError khi log emoji ra console
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Tự động tải và nạp FFmpeg/FFprobe vào Windows PATH (Sửa triệt để WinError 2 của Pydub/Whisper)
import static_ffmpeg
static_ffmpeg.add_paths()
# ------------------------

# --- MONKEY PATCH TORCHAUDIO DE GIA LAI FFMPEG / TORCHCODEC DLL ---
import torchaudio
import soundfile as sf
import torch

def patched_torchaudio_load(filepath, **kwargs):
    # Dùng soundfile thay cho libtorchcodec bị lỗi trên Windows do thiếu full-shared DLL
    data, sr = sf.read(filepath)
    if data.ndim == 1:
        data = data.reshape(-1, 1) # (frames, channels)
    data = data.T # (channels, frames)
    return torch.from_numpy(data.copy()).float(), sr

torchaudio.load = patched_torchaudio_load
# -----------------------------------------------------------------

# --- F5-TTS NATIVE IMPORT ---
from f5_tts.infer.utils_infer import (
    load_vocoder, 
    load_model, 
    infer_process, 
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav
)
from f5_tts.model import DiT
from omegaconf import OmegaConf
from importlib.resources import files

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
VOCAB_FILE = os.path.join(BASE_DIR, "F5-TTS-Vietnamese-ViVoice", "vocab.txt")
CKPT_FILE  = os.path.join(BASE_DIR, "F5-TTS-Vietnamese-ViVoice", "model_last.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# KHỞI TẠO MODEL 1 LẦN TRÊN GPU (Tránh tải lại mỗi khi chạy)
# --------------------------------------------------------
print("Loading model caching into GPU for fast inference...")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

# Lấy cài đặt kiến trúc của F5TTS_Base
model_cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_Base.yaml"))).model
model_cls = globals()[model_cfg.backbone]

# Load checkpoint (ViVoice)
ema_model = load_model(
    model_cls, 
    model_cfg.arch, 
    CKPT_FILE, 
    mel_spec_type="vocos", 
    vocab_file=VOCAB_FILE,
    device=device
)
print("Startup Complete. Model is ready on", device)

# --------------------------------------------------------
# 1. TRIM AUDIO MẪU
def preprocess_ref_audio(src_path, dst_path, max_sec=6):
    audio = AudioSegment.from_file(src_path)

    # Cắt bỏ khoảng lặng đầu/cuối
    chunks = silence.detect_nonsilent(audio, min_silence_len=200, silence_thresh=-40)
    if chunks:
        start_ms = max(0, chunks[0][0] - 100)
        end_ms   = min(len(audio), chunks[-1][1] + 100)
        audio    = audio[start_ms:end_ms]

    # Giới hạn độ dài tối đa
    audio = audio[:max_sec * 1000]

    # Chuẩn hóa âm lượng
    audio = audio.apply_gain(-audio.max_dBFS - 3)

    audio.export(dst_path, format="wav")
    duration = len(audio) / 1000
    return duration

# --------------------------------------------------------
# HÀM XỬ LÝ CHÍNH (Gradio endpoint)
def clone_voice(ref_audio, ref_text, gen_text, speed, progress=gr.Progress()):
    if ref_audio is None:
        return None, "❌ Vui lòng upload audio tham chiếu!"
    if not ref_text.strip():
        return None, "❌ QUAN TRỌNG: Bạn BẮT BUỘC phải điền Transcript/Nội dung của audio gốc vào ô trên để mô hình hoạt động trên Windows không lỗi!"
    if not gen_text.strip():
        return None, "❌ Vui lòng nhập văn bản muốn tổng hợp!"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    progress(0.1, desc="✂️ Đang chuẩn bị audio mẫu...")
    ref_path = os.path.join(OUTPUT_DIR, f"ref_{timestamp}.wav")
    try:
        duration = preprocess_ref_audio(ref_audio, ref_path, max_sec=6)
        # Format lại ref_audio và ref_text bằng tiền xử lý của F5-TTS
        ref_path, ref_text = preprocess_ref_audio_text(ref_path, ref_text, clip_short=False)
    except Exception as e:
        import traceback
        return None, f"❌ Tiền xử lý audio lỗi (bạn có nhớ phải import file .wav/mp3 thật sự không?):\n{traceback.format_exc()}"

    progress(0.3, desc=f"🚀 Suy luận nguyên bản trên GPU ({device})...")
    
    try:
        # F5-TTS infer_process tự động chunk văn bản và xử lý song song tối ưu trên tensor
        audio_segment, final_sample_rate, _ = infer_process(
            ref_path,
            ref_text,
            gen_text,
            ema_model,
            vocoder,
            mel_spec_type="vocos",
            speed=speed,
            cross_fade_duration=0.15,
            device=device
        )
    except Exception as e:
        import traceback
        return None, f"❌ Quá trình sinh giọng lỗi:\n{traceback.format_exc()}"

    progress(0.9, desc="💾 Lưu audio hoàn chỉnh...")
    final_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.wav")
    
    sf.write(final_path, audio_segment, final_sample_rate)
    
    # Cắt khoảng lặng rườm rà ở đuôi (nếu f5-tts có loop)
    remove_silence_for_generated_wav(final_path)

    # Dọn dẹp
    if os.path.exists(ref_path): os.remove(ref_path)

    progress(1.0, desc="✅ Hoàn tất!")

    if os.path.exists(final_path):
        ref_info = f" | Audio mẫu: {duration:.1f}s" if duration > 0 else ""
        return final_path, f"✅ Cực nhanh bằng Native Inference trên GPU! ({ref_info})"
    else:
        return None, "❌ Không tạo được file output."

def reset_form():
    return None, "", "", 1.0, None, ""

# --------------------------------------------------------
# GIAO DIỆN GRADIO
with gr.Blocks(title="🎙️ F5-TTS Vietnamese") as demo:

    gr.Markdown("""
    # 🎙️ F5-TTS Vietnamese — Clone Giọng Nói
    > **Model:** `hynt/F5-TTS-Vietnamese-ViVoice`
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📥 Đầu vào")
            ref_audio = gr.Audio(
                label="🎙️ Audio tham chiếu (tối đa 6 giây, 1 người nói, không nhạc nền)",
                type="filepath",
            )
            ref_text = gr.Textbox(
                label="📝 Transcript của audio tham chiếu (bắt buộc điền để tránh lặp)",
                placeholder="Nhập CHÍNH XÁC nội dung trong audio mẫu...",
                lines=2,
            )
            gr.Markdown("___")
            gen_text = gr.Textbox(
                label="💬 Văn bản muốn tổng hợp",
                placeholder="Nhập văn bản tiếng Việt đầy đủ dấu câu.\\nVí dụ: Xin chào, tôi là trợ lý AI.",
                lines=5,
            )
            speed = gr.Slider(
                minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                label="⚡ Tốc độ đọc",
                info="0.5 = chậm  |  1.0 = bình thường  |  2.0 = nhanh",
            )
            with gr.Row():
                btn_clone = gr.Button("🚀 Tạo giọng nói trực tiếp với Native GPU", variant="primary", scale=3)
                btn_reset = gr.Button("🔄 Reset", variant="secondary", scale=1)

        with gr.Column(scale=1):
            gr.Markdown("## 📤 Kết quả")
            out_audio = gr.Audio(
                label="🔊 Audio đã tổng hợp",
                type="filepath",
            )
            status_box = gr.Textbox(
                label="📊 Trạng thái",
                interactive=False,
                lines=3,
            )
            gr.Markdown("""
            ---
            ### 💡 Tính năng xử lý native
            - ✅ Tự động tận dụng VRAM GPU tạo giọng trơn tru, không phải gọi Subprocess
            - ✅ Xử lý chunk (chia câu) và ghép nối tự động mượt mà
            """)

    btn_clone.click(
        fn=clone_voice,
        inputs=[ref_audio, ref_text, gen_text, speed],
        outputs=[out_audio, status_box]
    )
    btn_reset.click(
        fn=reset_form,
        inputs=[],
        outputs=[ref_audio, ref_text, gen_text, speed, out_audio, status_box],
    )

if __name__ == "__main__":
    demo.launch(share=False, show_error=True, theme=gr.themes.Soft())
