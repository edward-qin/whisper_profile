import torch
import whisper

print("PyTorch version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0))

try:
    model = whisper.load_model("base")
    print("Whisper model loaded successfully!")
except Exception as e:
    print("Error loading Whisper model:", e)