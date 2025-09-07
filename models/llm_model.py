from copy import deepcopy
# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerateContentResponse
# noinspection PyPackageRequirements
from google.generativeai.types import Tool
from typing import Any, List, Union, Optional, Dict
from models.hf_model_wrapper import HFModelWrapper
from models.gemini_utils import (initialize_gemini_model,
                                 chat_to_gemini_format,
                                 get_gemini_models,
                                 )
from models.gemini_compatibility import MinGeminiCompatible, GeminiWrapper
from types import GeneratorType
import huggingface_hub


class LLMModel:
    def __init__(self, model_or_name: Union[str, genai.GenerativeModel, MinGeminiCompatible, Any],
                 *,
                 secret_token: Optional[str] = None,
                 system_instruction: Optional[str] = None,
                 tools: List[Tool] = None,
                 config: Optional[Dict[str, Any]] = None,
                 **generation_kwargs: Any
                 ):
        """
        Initialize the LLMModel with either a model name or an existing model instance.
        Args:
            model_or_name (Union[str, genai.GenerativeModel, MinGeminiCompatible, Any]):
                The model name as a string (for Gemini or Hugging Face) or an existing model instance.
                Any model that implements the MinGeminiCompatible interface can be used.
                In fact, it is duck typed, so if the model has the necessary methods, it will work.
            secret_token (Optional[str]): The secret token for authentication (Gemini API key or HF token).
            system_instruction (Optional[str]): The system instruction to guide the model's behavior.
            tools (List[Tool]): A list of tools to be used with the model (Gemini only).
            config (GenerationConfig): Predefined generation configuration for the model.
            **generation_kwargs: Additional generation parameters to override defaults in config.
        """

        self._model: Union[genai.GenerativeModel, MinGeminiCompatible, Any]
        self._secret_token: Optional[str] = secret_token
        self._is_logged_in: bool = False
        if isinstance(model_or_name, genai.GenerativeModel):
            # If an existing Gemini model wrap it with our interface
            self._model = GeminiWrapper(model_or_name)
        elif isinstance(model_or_name, MinGeminiCompatible):
            # If an existing MinGeminiCompatible model is provided, use it directly.
            self._model = model_or_name
        # String identifier cases
        elif isinstance(model_or_name, str):
            # If model name contains a '/' this is a Hugging Face model
            if "/" in model_or_name:
                # initialize HF wrapper
                try:
                    self._model = HFModelWrapper(model_or_name,
                                                 system_instruction=system_instruction,
                                                 hf_token=secret_token)
                    if secret_token is not None:
                        self._is_logged_in = True
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize Hugging Face model '{model_or_name}': {e}")
            elif model_or_name in get_gemini_models(secret_token=secret_token):
                # If a Gemini model name is provided, initialize the Gemini model.
                try:
                    model: GeminiWrapper = initialize_gemini_model(
                        model_name=model_or_name,
                        system_instruction=system_instruction,
                        google_secret=secret_token,
                        include_wrapper=True,
                    )
                    self._model = model
                    if secret_token is not None:
                        self._is_logged_in = True
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize Gemini model '{model_or_name}': {e}")
            else:
                raise ValueError(f"Invalid model name: {model_or_name}."
                                 f"Valid Gemini models are: {', '.join(get_gemini_models())}.")
        else:
            print("Warning: model_or_name is not a recognized type. Attempting to use it as-is.")
            self._model = model_or_name

        self._chat_session: Optional[Any] = None
        self._system_instruction: Optional[str] = system_instruction
        self._tools: List[Tool] = tools if tools is not None else []
        self._config: Dict[str, Any] = {}

        if not config and generation_kwargs:
            # Set up the config with any provided generation parameters
            config = dict(generation_kwargs)
        self._config = config

    def _is_gemini_model(self) -> bool:
        return isinstance(self._model, genai.GenerativeModel) or isinstance(self._model, GeminiWrapper)

    def login(self, secret_token: str):
        if self.has_secret_token:
            # Already logged in
            return

        if self._is_gemini_model():
            try:
                genai.configure(api_key=secret_token)
                self._secret_token = secret_token
                self._is_logged_in = True
            except Exception as e:
                raise RuntimeError(f"Failed to configure Gemini API with provided token: {e}")
        elif isinstance(self._model, HFModelWrapper):
            try:
                huggingface_hub.login(token=secret_token)
                self._secret_token = secret_token
                self._is_logged_in = True
            except Exception as e:
                raise RuntimeError(f"Failed to authenticate Hugging Face model with provided token: {e}")
        else:
            raise TypeError("Underlying model does not support login with a password/token.")

    @property
    def model_name(self) -> str:
        return self._model.model_name

    @property
    def system_instruction(self) -> Optional[str]:
        return self._system_instruction

    @property
    def tools(self) -> List[Tool]:
        return self._tools

    @property
    def has_secret_token(self) -> bool:
        return self._model and self._password is not None and len(self._password) > 0

    def has_token_changed(self, new_token: str) -> bool:
        return self._password != new_token

    # @staticmethod
    # def normalize_response(response):
    #     """Wraps Hugging Face generator and Gemini response into a unified generator of text chunks."""
    #     if isinstance(response, str):
    #         # Non-streaming response from HF or Gemini
    #         yield response
    #         return
    #
    #     if isinstance(response, GeneratorType):
    #         # Hugging Face: already a generator of strings
    #         yield from response
    #
    #     elif isinstance(response, GenerateContentResponse):
    #         # Gemini: stream across candidates/parts
    #         for candidate in response.candidates:
    #             for part in candidate.content.parts:
    #                 if hasattr(part, "text") and part.text:
    #                     yield part.text
    #     else:
    #         raise TypeError(f"Unsupported response type: {type(response)}")

    def generate_content(self,
                         message: str,
                         stream: bool = False,
                         tools: List[Tool] = None,
                         config: Optional[Dict[str, Any]] = None,
                         **generation_kwargs: Any
                         ) -> Union[GenerateContentResponse, GeneratorType, str]:

        if isinstance(self._model, MinGeminiCompatible) or hasattr(self._model, "generate_content"):
            # If the model is MinGeminiCompatible or has generate_content, use that method directly.
            response = self._model.generate_content(
                contents=message,
                generation_config=config if config is not None else self._config,
                tools=tools if tools is not None else self._tools,
                stream=stream,
                **generation_kwargs
            )
        else:
            # Otherwise raise an error
            raise TypeError("Underlying model does not support generate_content method.")

        return response

    def send_chat_message(self,
                          message: str,
                          chat_history: Optional[List[List[str]]] = None,
                          chat_session_reset: bool = False,
                          stream: bool = False,
                          tools: List[Tool] = None,
                          config: Optional[Dict[str, Any]] = None,
                          **generation_kwargs: Any
                          ) -> Union[GenerateContentResponse, GeneratorType, str]:

        formatted_chat_history: Optional[Union[List[Dict[str, Any]], List[List[str]]]] = None
        if chat_history is not None:
            if self._is_gemini_model():
                formatted_chat_history = chat_to_gemini_format(chat_history)
            else:
                formatted_chat_history = deepcopy(chat_history)

        if self._chat_session is None or chat_session_reset or chat_history is not None:
            # if the model has start_chat, call it; otherwise, for non-chat models we emulate one
            if hasattr(self._model, "start_chat") and callable(getattr(self._model, "start_chat")):
                self._chat_session = self._model.start_chat(history=formatted_chat_history)
            else:
                # fallback: create a dummy HFChatSession-like wrapper if possible
                raise RuntimeError("Underlying model does not support chat sessions.")

        # Merge generation_kwargs into GenerationConfig
        if not config:
            config = dict(generation_kwargs)
        else:
            config = {**config, **generation_kwargs}

        response = self._chat_session.send_message(
                                          message,
                                          tools=tools if tools is not None else self._tools,
                                          stream=stream,
                                          generation_config=config if config is not None else self._config,
                                          **generation_kwargs)

        return response

    def reset_chat(self):
        self._chat_session = None

    def update_system_instruction(self, new_instruction: str) -> None:
        self._system_instruction = new_instruction

        if self._is_gemini_model():
            # Recreate Gemini model with new system instruction (Gemini doesn't allow hot update)
            self._model = initialize_gemini_model(
                model_name=self._model.model_name,
                system_instruction=new_instruction,
                google_secret=self._password
            )

        elif isinstance(self._model, HFModelWrapper):
            # Hugging Face wrapper can be updated directly
            self._model.system_instruction = new_instruction

        # Reset chat so the new system instruction is applied fresh
        self.reset_chat()
