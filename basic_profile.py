import torch
import whisper

model_size = "medium"
audio_file = "long_speech.mp3"

log_dir = "./logs"
output_dir = "./profiling"
basic_file = f"{output_dir}/basic_profile_{model_size}.txt"

# Initialize PyTorch Profiler
model = whisper.load_model(model_size)
audio = whisper.load_audio(audio_file)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    # on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
    profile_memory=True,
    record_shapes=True,
    with_stack=True
) as prof:
    # Perform transcription
    result = model.transcribe(audio)

# Print summary table
with open(basic_file, "w") as f:
    f.write("SORT BY CUDA TIME\n")
    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    f.write("\nSORT BY CUDA MEMORY USAGE\n")
    f.write(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
