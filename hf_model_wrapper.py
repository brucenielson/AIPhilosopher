import torch
from typing import Any, List, Optional, Dict
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ----- Hugging Face wrapper: provides a minimal compatible interface -----
def _hf_gen_kwargs_from_config(config: Optional[GenerationConfig]) -> Dict[str, Any]:
    """Map a google GenerationConfig to HF pipeline kwargs (best-effort).
    Ensure a sensible default for max_new_tokens so transformers doesn't use max_length=20."""
    kw: Dict[str, Any] = {}
    if config is None:
        # default generation length if nothing provided
        kw["max_new_tokens"] = 256
        return kw

    if hasattr(config, "max_output_tokens") and getattr(config, "max_output_tokens") is not None:
        kw["max_new_tokens"] = int(getattr(config, "max_output_tokens"))
    if hasattr(config, "temperature") and getattr(config, "temperature") is not None:
        kw["temperature"] = float(getattr(config, "temperature"))
    if hasattr(config, "top_p") and getattr(config, "top_p") is not None:
        kw["top_p"] = float(getattr(config, "top_p"))
    if hasattr(config, "top_k") and getattr(config, "top_k") is not None:
        kw["top_k"] = int(getattr(config, "top_k"))
    if hasattr(config, "do_sample") and getattr(config, "do_sample") is not None:
        kw["do_sample"] = bool(getattr(config, "do_sample"))

    # fallback default if user didn't specify a token limit
    if "max_new_tokens" not in kw:
        kw["max_new_tokens"] = 256

    # if temperature explicitly 0, make deterministic
    if kw.get("temperature", None) == 0:
        kw["do_sample"] = False

    return kw


class HFModelWrapper:
    """
    Minimal wrapper around an HF text-generation pipeline that exposes:
     - generate_content(contents=..., generation_config=..., tools=..., stream=...)
     - start_chat(history=...) -> a session object with send_message(...)
    The returned objects have a `.text` attribute so existing code which uses
    getattr(response, "text", None) continues to work.
    """
    def __init__(self, model_name: str,
                 system_instruction: Optional[str] = None,
                 hf_token: Optional[str] = None):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.hf_token = hf_token
        # Check for GPU availability
        if torch.cuda.is_available():
            device = 0  # CUDA device index
            print("Using device: CUDA (GPU 0)")
            major, minor = torch.cuda.get_device_capability()
            if major < 7:
                print(f"GPU capability {major}.{minor} too old for Triton, disabling torch.compile.")
                torch._dynamo.config.disable = True
                torch._dynamo.config.suppress_errors = True
        else:
            device = -1  # CPU
            print("Using device: CPU")
        self.device = device

        # Load tokenizer and model with token
        # create tokenizer (we keep it to check token lengths)
        # use_fast=True for faster encoding if available
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            token=hf_token
        )

        # model_max_length may be > 1 or a very large number; None fallback handled later
        self.model_max_length = getattr(self.tokenizer, "model_max_length", None)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token
        ).cuda(device=device)

        # Now build pipeline with the loaded model + tokenizer
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )

    def generate_content(self, contents: str, generation_config: GenerationConfig = None, tools=None, stream=False):
        gen_kwargs = _hf_gen_kwargs_from_config(generation_config)

        # Ensure we avoid the transformers default max_length=20 issue:
        # - compute tokenized input length
        # - if input_len + max_new_tokens > model_max_length, then truncate prompt (tail) and/or reduce max_new_tokens
        try:
            enc = self.tokenizer(contents, return_tensors="pt", truncation=False)
            input_len = enc["input_ids"].shape[1]
            model_max = self.model_max_length if self.model_max_length and self.model_max_length > 0 else None

            requested_new = int(gen_kwargs.get("max_new_tokens", 256))

            if model_max is not None:
                available = model_max - input_len
                if available <= 0:
                    # prompt alone is longer than model's max. Truncate prompt to keep the tail
                    # choose keep_len to be model_max // 2 (arbitrary safe fallback) or model_max - 1
                    keep = max(1, model_max // 2)
                    # take last `keep` tokens
                    tail_ids = enc["input_ids"][0, -keep:]
                    contents = self.tokenizer.decode(tail_ids,
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
                    input_len = tail_ids.shape[0]
                    available = model_max - input_len

                # if requested would overflow, reduce it
                if requested_new > available:
                    gen_kwargs["max_new_tokens"] = max(1, available)
        except (ValueError, KeyError, IndexError, AttributeError):
            # if tokenizer fails for any reason, still ensure a default max_new_tokens is present
            gen_kwargs.setdefault("max_new_tokens", 256)

        # call pipeline (note: pass return_full_text only if you want the full concatenation)
        out = self.pipeline(contents, **gen_kwargs)
        # pipeline returns a list of dicts with "generated_text"
        text = out[0].get("generated_text", "")
        return text

    def start_chat(self, history: Optional[List[Dict[str, Any]]] = None):
        return HFChatSession(self, history or [])


class HFChatSession:
    """Keep a simple chat history and format a prompt for causal LMs."""
    def __init__(self, wrapper: HFModelWrapper, history: List[Dict[str, Any]]):
        self.wrapper = wrapper
        self.history = history  # list of dicts with 'role' and 'content'

    def _build_prompt(self, message: str) -> str:
        parts = []
        if self.wrapper.system_instruction:
            parts.append(f"[System]: {self.wrapper.system_instruction}")
        for item in self.history:
            role = item.get("role", "user").capitalize()
            parts.append(f"[{role}]: {item.get('content','')}")
        parts.append(f"[User]: {message}")
        parts.append("[Assistant]:")
        # join with newlines
        return "\n".join(parts)

    def send_message(self, message: str, generation_config: GenerationConfig = None, tools=None, stream=False):
        prompt = self._build_prompt(message)
        resp = self.wrapper.generate_content(prompt, generation_config=generation_config, tools=tools, stream=stream)
        # update history with user + assistant
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": resp})
        return resp
