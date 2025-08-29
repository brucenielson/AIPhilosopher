# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.api_core.exceptions import ResourceExhausted
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig, GenerateContentResponse
# noinspection PyPackageRequirements
from google.generativeai.types import Tool
from typing import Any, List, Union, Optional, Dict
import time
import re
import torch

# Optional HF imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# List of valid Gemini model_or_name names.
VALID_GEMINI_MODELS = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemma-3-27b-it",
    "gemma-3-8b-it",
    "gemma-3-8b-it-v1",
    "gemma-3-8b-it-v2"
]


def initialize_gemini_model(model_name: str = "gemini-2.0-flash",
                            system_instruction: Optional[str] = None,
                            google_secret: Optional[str] = None) -> genai.GenerativeModel:
    genai.configure(api_key=google_secret)
    if 'gemma' in model_name:
        # If using Gemma, set the system instruction to None as it does not support it.
        system_instruction = None

    model: genai.GenerativeModel = genai.GenerativeModel(
        model_name=model_name,  # gemini-2.0-flash-exp, gemini-2.0-flash, gemma-3-27b-it
        system_instruction=system_instruction
    )
    return model


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
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed. Install transformers[torch] and huggingface_hub.")
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.hf_token = hf_token
        # Check for GPU availability
        if torch.cuda.is_available():
            device = 0  # CUDA device index
            print("Using device: CUDA (GPU 0)")
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
        )

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
        except Exception:
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

# ---------------------------------------------------------------------


class LLMClient:
    def __init__(self, model_or_name: Union[str, genai.GenerativeModel, HFModelWrapper],
                 *,
                 secret_token: Optional[str] = None,
                 system_instruction: Optional[str] = None,
                 tools: List[Tool] = None,
                 config: GenerationConfig = None,
                 **generation_kwargs: Any
                 ):

        self._model: Union[genai.GenerativeModel, HFModelWrapper]
        # String identifier cases
        if isinstance(model_or_name, str):
            # If model name contains a '/' this is a Hugging Face model
            if "/" in model_or_name:
                if not HF_AVAILABLE:
                    raise RuntimeError("Hugging Face support requires `transformers` and `huggingface_hub` packages.")
                # initialize HF wrapper
                self._model = HFModelWrapper(model_or_name,
                                             system_instruction=system_instruction,
                                             hf_token=secret_token)
            elif model_or_name in VALID_GEMINI_MODELS:
                # If a Gemini model name is provided, initialize the Gemini model.
                self._model = initialize_gemini_model(
                    model_name=model_or_name,
                    system_instruction=system_instruction,
                    google_secret=secret_token
                )
            else:
                raise ValueError(f"Invalid model name: {model_or_name}. For Hugging Face models prefix with 'hf:'. "
                                 f"Valid Gemini models are: {', '.join(VALID_GEMINI_MODELS)}.")
        elif isinstance(model_or_name, genai.GenerativeModel):
            self._model = model_or_name
        elif HF_AVAILABLE and isinstance(model_or_name, HFModelWrapper):
            self._model = model_or_name
        else:
            raise TypeError("model_or_name must be a string, an instance of genai.GenerativeModel, or HFModelWrapper.")

        self._chat_session: Optional[Any] = None
        self._system_instruction: Optional[str] = system_instruction
        self._password: Optional[str] = secret_token
        self._tools: List[Tool] = tools if tools is not None else []
        self._config: Optional[GenerationConfig] = None

        if secret_token and isinstance(model_or_name, genai.GenerativeModel):
            # Login to the Gemini API using the provided secret_token.
            genai.configure(api_key=secret_token)

        if config is None and generation_kwargs:
            # Set up the config with any provided generation parameters
            config = GenerationConfig(**generation_kwargs)
        self._config = config

    def login(self, password: str):
        if isinstance(self._model, genai.GenerativeModel):
            genai.configure(api_key=password)
            self._password = password
        elif HF_AVAILABLE and isinstance(self._model, HFModelWrapper):
            # For HF, store token and re-create pipeline if desired.
            self._model.hf_token = password
            # NOTE: pipeline re-creation might be necessary depending on auth scope.
            # self._model = HFModelWrapper(self._model.model_name,
            #                              system_instruction=self._model.system_instruction,
            #                              hf_token=password)
            self._password = password

    def generate_content(self,
                         message: str,
                         stream: bool = False,
                         tools: List[Tool] = None,
                         config: GenerationConfig = None,
                         **generation_kwargs: Any
                         ) -> str:

        return LLMClient._send_message(self._model,
                                       message,
                                       tools=tools if tools is not None else self._tools,
                                       stream=stream,
                                       config=config if config is not None else self._config,
                                       **generation_kwargs)

    def send_chat_message(self,
                          message: str,
                          chat_history: Optional[List[Dict[str, Any]]] = None,
                          chat_session_reset: bool = False,
                          stream: bool = False,
                          tools: List[Tool] = None,
                          config: GenerationConfig = None,
                          **generation_kwargs: Any
                          ) -> Union[GenerateContentResponse, str]:

        if self._chat_session is None or chat_session_reset:
            # if the model has start_chat, call it; otherwise, for non-chat models we emulate one
            if hasattr(self._model, "start_chat") and callable(getattr(self._model, "start_chat")):
                self._chat_session = self._model.start_chat(history=chat_history)
            else:
                # fallback: create a dummy HFChatSession-like wrapper if possible
                raise RuntimeError("Underlying model does not support chat sessions.")

        return LLMClient._send_message(self._chat_session,
                                       message,
                                       tools=tools if tools is not None else self._tools,
                                       stream=stream,
                                       config=config if config is not None else self._config,
                                       **generation_kwargs)

    def reset_chat(self):
        self._chat_session = None

    def update_system_instruction(self, new_instruction: str) -> None:
        self._system_instruction = new_instruction

        if isinstance(self._model, genai.GenerativeModel):
            # Recreate Gemini model with new system instruction (Gemini doesn't allow hot update)
            self._model = initialize_gemini_model(
                model_name=self._model.model_name,
                system_instruction=new_instruction,
                google_secret=self._password
            )

        elif HF_AVAILABLE and isinstance(self._model, HFModelWrapper):
            # Hugging Face wrapper can be updated directly
            self._model.system_instruction = new_instruction

        # Reset chat so the new system instruction is applied fresh
        self.reset_chat()

    # Gemini/HF specific utility methods
    @staticmethod
    def _extract_retry_seconds(exc: ResourceExhausted, default: int = 15) -> int:
        """
        Extracts retry_delay.seconds from the exception's details text.
        Falls back to `default` if not found or parsing fails.
        """
        try:
            details = str(getattr(exc, "details", ""))
            match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', details)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return default

    @staticmethod
    def _send_message(model: Any,
                      message: str,
                      tools: List[Tool] = None,
                      stream: bool = False,
                      config: GenerationConfig = None,
                      **generation_kwargs: Any) -> Union[GenerateContentResponse, str]:

        if config is None and generation_kwargs:
            config = GenerationConfig(**generation_kwargs)

        try:
            # Duck-typed chat detection (works for Google ChatSession and HFChatSession)
            if hasattr(model, "send_message") and callable(getattr(model, "send_message")):
                return model.send_message(message,
                                          generation_config=config,
                                          tools=tools,
                                          stream=stream)
            # else try generate_content for model wrappers (Gemini or HFModelWrapper)
            elif hasattr(model, "generate_content") and callable(getattr(model, "generate_content")):
                response = model.generate_content(
                    contents=message,
                    generation_config=config,
                    tools=tools,
                    stream=stream
                )
                return getattr(response, "text", None) or "[No response text]"
            else:
                raise TypeError("Provided model object does not implement send_message or generate_content.")
        except ResourceExhausted as e:
            # Handle Google rate limit errors (Gemini)
            delay = LLMClient._extract_retry_seconds(e)
            if delay is None or delay <= 0:
                delay = 15
            print(f"\nRate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            return LLMClient._send_message(model,
                                           message,
                                           tools=tools,
                                           stream=stream,
                                           config=config,
                                           **generation_kwargs)
        except Exception as e:
            # A simple retry heuristic for HF rate-limit-style errors
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                # small backoff and retry once
                delay = 10
                print(f"Rate limit-like error detected from HF/provider. Retrying in {delay} seconds...")
                time.sleep(delay)
                return LLMClient._send_message(model,
                                               message,
                                               tools=tools,
                                               stream=stream,
                                               config=config,
                                               **generation_kwargs)
            print(f"Error during chat message sending: {e}")
            raise
