from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading
import torch

torch_dtype = torch.float32

device = "auto"
if device == "auto" and torch.cuda.is_available():
    device = "cuda"
    print("Using device: CUDA (GPU)")
    major, minor = torch.cuda.get_device_capability()
    if major < 7:
        print(f"GPU capability {major}.{minor} too old for Triton, disabling torch.compile.")
        # noinspection PyProtectedMember
        torch._dynamo.config.disable = True
        # noinspection PyProtectedMember
        torch._dynamo.config.suppress_errors = True
else:
    device = "cpu"
    print("Using device: CPU")

# model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
tok = AutoTokenizer.from_pretrained(
    "google/gemma-3-270m",
    use_fast=True
)
# tok = AutoTokenizer.from_pretrained("google/gemma-3-270m")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m",
    torch_dtype=torch_dtype,
).to(device=device)

inputs = tok("Hello, my name is", return_tensors="pt").to(model.device)
# streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
streamer = TextIteratorStreamer(tok, skip_special_tokens=True)

thread = threading.Thread(
    target=model.generate,
    kwargs=dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=50,
        do_sample=False,
    ),
)
thread.start()

for new_text in streamer:
    print(new_text, end="", flush=True)
