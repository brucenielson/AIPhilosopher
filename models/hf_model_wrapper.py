import torch
from typing import List, Optional, Union, Generator
# # noinspection PyPackageRequirements
# from google.generativeai.types.generation_types import GenerationConfig as GGenConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.generation import GenerationConfig as HFGenConfig
import threading
from models.gemini_compatibility import GeminiCompatible, GeminiChatSessionCompatible
from huggingface_hub import HfFolder
from utilities.general_utils import logger

TRITON_REQUIRED_CAPABILITY = 7  # minimum GPU capability for triton backend

DECODING_PRESETS = {
    "balanced": dict(
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=200,
    ),
    "creative": dict(
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.0,
        max_new_tokens=300,
    ),
    "factual": dict(
        do_sample=False,   # greedy decoding
        max_new_tokens=200,
    ),
    "diverse": dict(
        do_sample=True,
        temperature=1.2,
        top_p=0.95,
        top_k=100,
        repetition_penalty=1.15,
        max_new_tokens=250,
    ),
    "short_answer": dict(
        do_sample=True,
        temperature=0.7,
        top_p=0.85,
        max_new_tokens=50,
    ),
}


# ----- Hugging Face wrapper: provides a minimal compatible interface -----
class HFModelWrapper(GeminiCompatible):
    """
    Minimal wrapper around an HF text-generation pipeline that exposes:
     - generate_content(contents=..., generation_config=..., tools=..., stream=...)
     - start_chat(history=...) -> a session object with send_message(...)
    The returned objects have a `.text` attribute so existing code which uses
    getattr(response, "text", None) continues to work.
    """
    def __init__(
            self,
            model_name: str,
            system_instruction: Optional[str] = None,
            hf_token: Optional[str] = None,
            device: str = "auto",
            dtype: Union[str, torch.dtype] = "auto",
    ):
        super().__init__(model_name, system_instruction)
        self._model_name = model_name
        self._system_instruction = system_instruction
        if hf_token is None:
            hf_token = HfFolder.get_token()
        self._hf_token = hf_token
        self._dtype = dtype
        self._device = device
        self._tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        self._model: PreTrainedModel
        self._model_max_length: int = 1024  # will be updated after loading model

        # Pick dtype (always 32 bit for now)
        torch_dtype: Optional[torch.dtype] = torch.float32
        # if dtype == "auto":
        #     torch_dtype = torch.float16 if torch.cuda.is_available() else None
        # elif isinstance(dtype, str):
        #     torch_dtype = getattr(torch, dtype)
        # else:
        #     torch_dtype = dtype

        # Check for GPU availability
        if self._device == "auto" and torch.cuda.is_available():
            self._device = "cuda"
            logger.warning("Using device: CUDA (GPU)")
            major, minor = torch.cuda.get_device_capability()
            if major < TRITON_REQUIRED_CAPABILITY:
                logger.warning(f"GPU capability {major}.{minor} too old for Triton, disabling torch.compile.")
                # noinspection PyProtectedMember
                torch._dynamo.config.disable = True
                # noinspection PyProtectedMember
                torch._dynamo.config.suppress_errors = True
        else:
            self._device = "cpu"
            logger.warning("Using device: CPU")

        # Load tokenizer and model with token
        # create tokenizer (we keep it to check token lengths)
        # use_fast=True for faster encoding if available
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            use_fast=True,
            token=self._hf_token
        )

        # model_max_length may be > 1 or a very large number; None fallback handled later
        # self._model_max_length: int = getattr(self._tokenizer, "model_max_length", None)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=self._hf_token,
            torch_dtype=torch_dtype,
        ).to(device=self._device)

        # Now build pipeline with the loaded model + tokenizer
        # self._pipeline = pipeline(
        #     "text-generation",
        #     model=self._model,
        #     tokenizer=self._tokenizer,
        #     use_fast=True,
        # )

    @property
    def is_logged_in(self) -> bool:
        return self._hf_token is not None and self._hf_token == HfFolder.get_token()

    @property
    def system_instruction(self) -> Optional[str]:
        return self._system_instruction

    @system_instruction.setter
    def system_instruction(self, value: Optional[str]):
        self._system_instruction = value

    def generate_content(self,
                         contents: str,
                         generation_config: Optional[HFGenConfig] = None,
                         tools=None,
                         stream=False) -> str | Generator[str, None, None]:
        if generation_config is None:
            gen_kwargs: dict = {}
        elif isinstance(generation_config, HFGenConfig):
            gen_kwargs = vars(generation_config).copy()
        elif isinstance(generation_config, dict):
            gen_kwargs = generation_config.copy()
        else:
            raise TypeError(f"Unsupported generation_config type: {type(generation_config)}")

        gen_kwargs.update(dict(do_sample=True, num_beams=1))

        # Respect model’s max length (don’t hardcode)
        self._model_max_length = getattr(
            self._model.config, "max_position_embeddings", 1024
        )

        inputs: dict[str, torch.Tensor] = self._tokenizer(
            contents,
            return_tensors="pt",
            truncation=True,
            max_length=self._model_max_length,
        ).to(self._model.device)

        ids = inputs["input_ids"]
        logger.warning("max token id: %d", ids.max().item())
        logger.warning("vocab size: %d", self._model.config.vocab_size)

        logger.warning("model max positions: %d", self._model.config.max_position_embeddings)
        logger.warning("input length: %d", inputs["input_ids"].shape[1])

        if not stream:
            outputs: torch.LongTensor = self._model.generate(**inputs, **gen_kwargs)
            text: str = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text[len(contents):]  # strip input prompt from start of output
        else:
            # --- Streaming branch ---
            # TODO: Reuse gen_kwargs from above instead of resetting it
            # gen_kwargs.setdefault("max_new_tokens", 256)

            # Decode canonical prompt for stripping later
            canonical_prompt = self._tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=True
            )

            # Create streamer
            streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True)

            # Launch generation in background thread
            thread = threading.Thread(
                target=self._model.generate,
                kwargs=dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                ),
            )
            thread.start()

            def stream_generator():
                buffer = ""
                seen_prompt = False

                for chunk in streamer:
                    if not chunk:
                        continue

                    if not seen_prompt:
                        buffer += chunk
                        idx = buffer.find(canonical_prompt)
                        if idx != -1:
                            start = idx + len(canonical_prompt)
                            remaining = buffer[start:]
                            seen_prompt = True
                            buffer = ""
                            yield remaining
                    else:
                        yield chunk

                thread.join()  # ensure generation finished

            return stream_generator()

        # def stream_generator():
        #     generated_tokens = 0
        #     for new_text in streamer:
        #         # Count tokens in this chunk
        #         chunk_len = len(self._tokenizer(new_text, add_special_tokens=False)["input_ids"])
        #         generated_tokens += chunk_len
        #         # Only yield tokens after the prompt
        #         if generated_tokens > prompt_len:
        #             # trim the first (prompt_len) tokens from the first chunk
        #             if generated_tokens - chunk_len < prompt_len:
        #                 # how many tokens to skip in this chunk
        #                 skip_tokens = prompt_len - (generated_tokens - chunk_len)
        #                 # decode the remaining tokens
        #                 remaining_text_ids = self._tokenizer(new_text, add_special_tokens=False)["input_ids"][
        #                                      skip_tokens:]
        #                 yield self._tokenizer.decode(remaining_text_ids, skip_special_tokens=True)
        #             else:
        #                 yield new_text

        # Ensure we avoid the transformers default max_length=20 issue:
        # - compute tokenized input length
        # - if input_len + max_new_tokens > model_max_length, then truncate prompt (tail) and/or reduce max_new_tokens
        # try:
        #     enc = self.tokenizer(contents, return_tensors="pt", truncation=False)
        #     input_len = enc["input_ids"].shape[1]
        #     model_max = self.model_max_length if self.model_max_length and self.model_max_length > 0 else None
        #
        #     requested_new = int(gen_kwargs.get("max_new_tokens", 256))
        #
        #     if model_max is not None:
        #         available = model_max - input_len
        #         if available <= 0:
        #             # prompt alone is longer than model's max. Truncate prompt to keep the tail
        #             # choose keep_len to be model_max // 2 (arbitrary safe fallback) or model_max - 1
        #             keep = max(1, model_max // 2)
        #             # take last `keep` tokens
        #             tail_ids = enc["input_ids"][0, -keep:]
        #             contents = self.tokenizer.decode(tail_ids,
        #                                              skip_special_tokens=True,
        #                                              clean_up_tokenization_spaces=True)
        #             input_len = tail_ids.shape[0]
        #             available = model_max - input_len
        #
        #         # if requested would overflow, reduce it
        #         if requested_new > available:
        #             gen_kwargs["max_new_tokens"] = max(1, available)
        # except (ValueError, KeyError, IndexError, AttributeError):
        #     # if tokenizer fails for any reason, still ensure a default max_new_tokens is present
        #     gen_kwargs.setdefault("max_new_tokens", 256)
        #
        # # call pipeline (note: pass return_full_text only if you want the full concatenation)
        # out = self._pipeline(contents, return_full_text=False, **gen_kwargs)
        # # pipeline returns a list of dicts with "generated_text"
        # text = out[0].get("generated_text", "")
        # return text

    def start_chat(self, history: Optional[List[List[str]]] = None) -> GeminiChatSessionCompatible:
        return HFChatSession(self, history or [])


class HFChatSession(GeminiChatSessionCompatible):
    """Keep a simple chat history and format a prompt for causal LMs."""
    def __init__(self, wrapper: HFModelWrapper, history: List[List[str]]):
        super().__init__(history)
        self._wrapper: HFModelWrapper = wrapper

    def _build_prompt(self, message: str) -> str:
        parts = []
        if self._wrapper.system_instruction:
            parts.append(f"[System]: {self._wrapper.system_instruction}")
        for item in self._history:
            parts.append(f"[User]: {item[0]}")
            parts.append(f"[Assistant]: {item[1]}")
        parts.append(f"[User]: {message}")
        parts.append("[Assistant]:")
        # join with newlines
        return "\n".join(parts)

    def send_message(self, content: str, generation_config: Optional[HFGenConfig] = None, tools=None, stream=False):
        prompt = self._build_prompt(content)
        resp = self._wrapper.generate_content(prompt, generation_config=generation_config, tools=tools, stream=stream)

        if isinstance(resp, Generator):
            # Add a placeholder in history
            self._history.append([content, ""])

            # Wrap the generator to update history incrementally
            def history_stream_wrapper(gen):
                accumulated = ""
                for chunk in gen:
                    accumulated += chunk
                    # Update last history entry with accumulated text
                    self._history[-1][1] = accumulated
                    yield chunk

            return history_stream_wrapper(resp)
        elif isinstance(resp, str):
            # Non-streaming: just store the response in history
            self._history.append([content, resp])
            return resp
        else:
            raise ValueError("Unexpected response type from generate_content")
