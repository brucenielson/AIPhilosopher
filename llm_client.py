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


# Code to initialize the Gemini model with optional system instruction and Google secret key.
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


class LLMClient:
    def __init__(self, model: Union[genai.GenerativeModel],
                 *,
                 password: Optional[str] = None,
                 system_instruction: Optional[str] = None,
                 tools: List[Tool] = None,
                 config: GenerationConfig = None,
                 **generation_kwargs: Any
                 ):

        self._model: genai.GenerativeModel = model
        self._chat_session: Optional[ChatSession] = None
        self._system_instruction: Optional[str] = system_instruction
        self._password: Optional[str] = password
        self._tools: List[Tool] = tools if tools is not None else []
        self._config: Optional[GenerationConfig] = None

        if password and isinstance(model, genai.GenerativeModel):
            # Login to the Gemini API using the provided password.
            genai.configure(api_key=password)

        if config is None and generation_kwargs:
            # Set up the config with any provided generation parameters
            config = GenerationConfig(**generation_kwargs)
            self._config: Optional[GenerationConfig] = config

    def login(self, password: str):
        if isinstance(self._model, genai.GenerativeModel):
            genai.configure(api_key=password)
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
            self._chat_session = self._model.start_chat(history=chat_history)

        return LLMClient._send_gemini_message(self._chat_session,
                                              message,
                                              tools=tools if tools is not None else self._tools,
                                              stream=stream,
                                              config=config if config is not None else self._config,
                                              **generation_kwargs)

    def reset_chat(self):
        if self._chat_session is not None:
            self._chat_session = None

    # Gemini specific static methods
    @staticmethod
    def _extract_retry_seconds(exc: ResourceExhausted, default: int = 15) -> int:
        """
        Extracts retry_delay.seconds from the exception's details text.
        Falls back to `default` if not found or parsing fails.
        """
        # noinspection PyBroadException
        try:
            details = str(getattr(exc, "details", ""))
            match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', details)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return default

    @staticmethod
    def _send_gemini_message(model: Union[ChatSession, genai.GenerativeModel],
                             message: str,
                             tools: List[Tool] = None,
                             stream: bool = False,
                             config: GenerationConfig = None,
                             **generation_kwargs: Any) -> Union[GenerateContentResponse, str]:

        if config is None and generation_kwargs:
            # Set up the config with any provided generation parameters
            config: GenerationConfig = GenerationConfig(**generation_kwargs)

        try:
            if isinstance(model, ChatSession):
                # If tools are provided, include them in the message
                return model.send_message(message,
                                          generation_config=config,
                                          tools=tools,
                                          stream=stream)
            else:
                # Otherwise, generate content directly without a chat session
                response = model.generate_content(
                    contents=message,
                    generation_config=config,
                    tools=tools,
                    stream=stream
                )
                return getattr(response, "text", None) or "[No response text]"

        except ResourceExhausted as e:
            # Handle rate limit errors by checking for retry_delay
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
            print(f"Error during chat message sending: {e}")
            raise
