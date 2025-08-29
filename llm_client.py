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
from hf_model_wrapper import HFModelWrapper, HF_AVAILABLE
from gemini_utils import initialize_gemini_model, VALID_GEMINI_MODELS


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
        except (ValueError, AttributeError):
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
