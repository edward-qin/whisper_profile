import torch
import whisper

# Initialize PyTorch Profiler
model = whisper.load_model("base")
audio = whisper.load_audio("long_speech.mp3")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs"),
    record_shapes=True,
    with_stack=True
) as prof:
    # Perform transcription
    result = model.transcribe(audio)

# Print summary table
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
