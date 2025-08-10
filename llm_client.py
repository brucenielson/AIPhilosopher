# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.api_core.exceptions import ResourceExhausted
# noinspection PyPackageRequirements
from google.generativeai import ChatSession
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig, GenerateContentResponse
# noinspection PyPackageRequirements
from google.generativeai.types import Tool
from typing import Any, List, Union, Optional, Dict
import time
import re

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
    """Map a google GenerationConfig to HF pipeline kwargs (best-effort)."""
    if config is None:
        return {}

    kw: Dict[str, Any] = {}
    # Best-effort mappings (names may differ across libs)
    if hasattr(config, "max_output_tokens"):
        kw["max_new_tokens"] = int(getattr(config, "max_output_tokens"))
    if hasattr(config, "temperature"):
        kw["temperature"] = float(getattr(config, "temperature"))
    if hasattr(config, "top_p"):
        kw["top_p"] = float(getattr(config, "top_p"))
    if hasattr(config, "top_k"):
        kw["top_k"] = int(getattr(config, "top_k"))
    if hasattr(config, "do_sample"):
        kw["do_sample"] = bool(getattr(config, "do_sample"))
    # fallback: if no sampling settings and temperature==0, make deterministic
    if "temperature" in kw and kw["temperature"] == 0:
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
    def __init__(self, model_name: str, system_instruction: Optional[str] = None,
                 hf_token: Optional[str] = None, device: int = -1):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not installed. Install transformers[torch] and huggingface_hub.")
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.hf_token = hf_token
        # device -1 => CPU, >=0 => CUDA device id
        self.device = device
        # create pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device,
            use_auth_token=hf_token
        )

    def generate_content(self, contents: str, generation_config: GenerationConfig = None, tools=None, stream=False):
        gen_kwargs = _hf_gen_kwargs_from_config(generation_config)
        # pipeline returns list[dict] with 'generated_text'
        out = self.pipeline(contents, **gen_kwargs)
        text = out[0].get("generated_text", "")
        class Resp:
            pass
        r = Resp()
        r.text = text
        return r

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
        self.history.append({"role": "assistant", "content": resp.text})
        return resp

# ---------------------------------------------------------------------


class LLMClient:
    def __init__(self, model_or_name: Union[str, genai.GenerativeModel, HFModelWrapper],
                 *,
                 secret_token: Optional[str] = None,
                 system_instruction: Optional[str] = None,
                 tools: List[Tool] = None,
                 config: GenerationConfig = None,
                 hf_device: int = -1,
                 **generation_kwargs: Any
                 ):

        self._model: Union[genai.GenerativeModel, HFModelWrapper]
        # String identifier cases
        if isinstance(model_or_name, str):
            # Hugging Face string prefix: 'hf:MODEL_ID' (convention used here)
            if model_or_name.startswith("hf:") or model_or_name.startswith("huggingface:"):
                if not HF_AVAILABLE:
                    raise RuntimeError("Hugging Face support requires `transformers` and `huggingface_hub` packages.")
                model_id = model_or_name.split(":", 1)[1]
                # initialize HF wrapper
                self._model = HFModelWrapper(model_id, system_instruction=system_instruction, hf_token=secret_token, device=hf_device)
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
            # self._model = HFModelWrapper(self._model.model_name, system_instruction=self._model.system_instruction, hf_token=password, device=self._model.device)
            self._password = password

    def generate_content(self,
                         message: str,
                         stream: bool = False,
                         tools: List[Tool] = None,
                         config: GenerationConfig = None,
                         **generation_kwargs: Any
                         ) -> str:

        return LLMClient._send_gemini_message(self._model,
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
                          ) -> GenerateContentResponse:

        if self._chat_session is None or chat_session_reset:
            # if the model has start_chat, call it; otherwise, for non-chat models we emulate one
            if hasattr(self._model, "start_chat") and callable(getattr(self._model, "start_chat")):
                self._chat_session = self._model.start_chat(history=chat_history)
            else:
                # fallback: create a dummy HFChatSession-like wrapper if possible
                raise RuntimeError("Underlying model does not support chat sessions.")

        return LLMClient._send_gemini_message(self._chat_session,
                                              message,
                                              tools=tools if tools is not None else self._tools,
                                              stream=stream,
                                              config=config if config is not None else self._config,
                                              **generation_kwargs)

    def reset_chat(self):
        self._chat_session = None

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
    def _send_gemini_message(model: Any,
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
            return LLMClient._send_gemini_message(model,
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
                return LLMClient._send_gemini_message(model,
                                                      message,
                                                      tools=tools,
                                                      stream=stream,
                                                      config=config,
                                                      **generation_kwargs)
            print(f"Error during chat message sending: {e}")
            raise
