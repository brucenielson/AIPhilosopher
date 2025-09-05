# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.auth.exceptions import DefaultCredentialsError
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Callable, Iterable
import re
import time
# noinspection PyPackageRequirements
from google.api_core.exceptions import ResourceExhausted
# noinspection PyPackageRequirements
from google.genai.types import Content
# noinspection PyPackageRequirements
from google.generativeai.types import (
    content_types,
    generation_types,
    safety_types,
    helper_types,
    GenerateContentResponse,
)
# noinspection PyPackageRequirements
from google.generativeai import ChatSession

# Useful links on Gemini:
# https://medium.com/%40adarsh.ajay/unleashing-the-power-of-google-gemini-with-python-a-step-by-step-guide-ed5e2ea1818f

# List of valid Gemini model_or_name names.
VALID_GEMINI_MODELS = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemma-3-27b-it",
    "gemma-3-8b-it",
    "gemma-3-8b-it-v1",
    "gemma-3-8b-it-v2"
]


def get_gemini_models(secret_token: Optional[str] = None) -> List[str]:
    try:
        if secret_token:
            genai.configure(api_key=secret_token)
        return [model.name.removeprefix("models/") for model in genai.list_models()]
    except DefaultCredentialsError:
        return VALID_GEMINI_MODELS


# Taken from https://medium.com/latinxinai/simple-chatbot-gradio-google-gemini-api-4ce02fbaf09f
def chat_to_gemini_format(history: List[List[str]]) -> List[Dict[str, Any]]:
    new_history = []
    for chat_response in history:
        new_history.append({"parts": [{"text": chat_response[0]}], "role": "user"})
        new_history.append({"parts": [{"text": chat_response[1]}], "role": "model"})
    return new_history


def gemini_extract_retry_seconds(exc: ResourceExhausted, default: int = 15) -> int:
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


def with_retry(fn: Callable, *args, max_retries: int = 5, **kwargs) -> Any:
    """
    Call a function with retry handling for Gemini and HF-like rate-limit errors.
    Retries up to `max_retries` times before raising.
    """

    attempts = 0
    while attempts <= max_retries:
        try:
            return fn(*args, **kwargs)

        except ResourceExhausted as e:
            delay = gemini_extract_retry_seconds(e) or 15
            print(f"\nGemini Rate limit exceeded. Retrying in {delay} seconds... (attempt {attempts+1})")
            time.sleep(delay)

        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                delay = 15
                print(f"Rate limit-like error detected from provider. "
                      f"Retrying in {delay} seconds... (attempt {attempts+1})")
                time.sleep(delay)
            else:
                print(f"Error during chat message sending: {e}")
                raise

        attempts += 1

    raise RuntimeError(f"Max retries exceeded ({max_retries}) for {fn.__name__}")


def retryable(max_retries: int = 5):
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            return with_retry(fn, *args, max_retries=max_retries, **kwargs)
        return wrapper
    return decorator


# This class is not strictly necessary. LLMModel can use Gemini directly.
# However, it is useful to have a minimal Gemini-like interface that other providers can implement.
# This allows for easier switching between providers if needed.
class MinGeminiCompatible(ABC):
    """
    Abstract Gemini-like interface that other providers must implement.
    """

    def __init__(self, model: Any,
                 model_name: Optional[str] = None,
                 system_instruction: Optional[str] = None,
                 secret_token: Optional[str] = None):
        self._model: genai.GenerativeModel = model
        self._model_name: str = model_name
        self._system_instruction: str = system_instruction
        self._secret_token: str = secret_token

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system_instruction(self) -> Optional[str]:
        return self._system_instruction

    @abstractmethod
    def generate_content(self, contents: str, **kwargs) -> str:
        """
        Generate text for a single prompt.
        """
        pass

    @abstractmethod
    def start_chat(self, history: List[Dict[str, Any]]):
        """
        Start a chat session with an optional history.
        Should return an object that has .send_message(prompt) -> str
        """
        pass


class GeminiChatSessionCompatible(ABC):
    """
    Gemini-compatible chat session interface.
    Mimics the Google SDK `start_chat` object.
    """

    def __init__(self, history: Optional[Union[List[List[str]], List[Dict[str, Any]]]] = None):
        self._history: Union[List[List[str]], Optional[List[Dict[str, Any]]]] = history or []

    @abstractmethod
    def send_message(self, contents: str, **kwargs) -> str:
        """
        Send a message in the ongoing chat and return the model's reply.
        """
        pass

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full chat history in Gemini-like format."""
        return self._history


class GeminiChatSessionWrapper(GeminiChatSessionCompatible):
    def __init__(self, chat_session):
        super().__init__(history=None)
        self._chat_session: ChatSession = chat_session

    @retryable(max_retries=5)
    def send_message(
            self,
            content: Union[content_types.ContentType, str],
            *,
            generation_config: Optional[generation_types.GenerationConfigType] = None,
            safety_settings: Optional[safety_types.SafetySettingOptions] = None,
            stream: bool = False,
            tools: Optional[content_types.FunctionLibraryType] = None,
            tool_config: Optional[content_types.ToolConfigType] = None,
            request_options: Optional[helper_types.RequestOptionsType] = None,
    ) -> GenerateContentResponse:
        """
        Wrapper around Gemini's ChatSession.send_message with retry logic.
        Mirrors the official Google signature.
        """
        if isinstance(content, str):
            content = content_types.to_content(content)

        return self._chat_session.send_message(
            content,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream,
            tools=tools,
            tool_config=tool_config,
            request_options=request_options,
        )

    def get_history(self) -> List[Content]:
        # override to return SDK-tracked history
        return list(self._chat_session.history)

    def __getattr__(self, name: str):
        # delegate any missing methods to the underlying Gemini chat object
        return getattr(self._chat, name)


class GeminiWrapper(MinGeminiCompatible):
    """
    Gemini wrapper that is compatible with MinGeminiCompatible but still
    exposes the full underlying GenerativeModel API transparently.
    """

    def __init__(self, model: genai.GenerativeModel):
        super().__init__(model=model,
                         model_name=model.model_name,
                         system_instruction=None,
                         secret_token=None)

    @retryable(max_retries=5)
    def generate_content(
            self,
            contents: content_types.ContentsType,
            *,
            generation_config: Optional[generation_types.GenerationConfigType] = None,
            safety_settings: Optional[safety_types.SafetySettingOptions] = None,
            stream: bool = False,
            tools: Optional[content_types.FunctionLibraryType] = None,
            tool_config: Optional[content_types.ToolConfigType] = None,
            request_options: Optional[helper_types.RequestOptionsType] = None,
    ) -> GenerateContentResponse:
        return self._model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream,
            tools=tools,
            tool_config=tool_config,
            request_options=request_options,
        )

    def start_chat(
            self,
            *,
            history: Optional[Iterable[content_types.StrictContentType]] = None,
            enable_automatic_function_calling: bool = False,
    ) -> GeminiChatSessionWrapper:
        """
        Wrapper around Gemini's start_chat.
        Mirrors the official signature.
        """
        # TODO: Do conversion of history if needed
        chat = self._model.start_chat(
            history=history,
            enable_automatic_function_calling=enable_automatic_function_calling,
        )
        return GeminiChatSessionWrapper(chat)  # wraps in your BaseChatSession

    @property
    def model_name(self) -> str:
        return self._model.model_name

    # --- transparent forwarding ---
    def __getattr__(self, name: str):
        """
        If the attribute is not found on this wrapper,
        delegate it to the underlying GenerativeModel.
        """
        return getattr(self._model, name)


def initialize_gemini_model(model_name: str = "gemini-2.0-flash",
                            system_instruction: Optional[str] = None,
                            google_secret: Optional[str] = None,
                            include_wrapper: bool = True) -> Union[genai.GenerativeModel, GeminiWrapper]:
    genai.configure(api_key=google_secret)
    if 'gemma' in model_name:
        # If using Gemma, set the system instruction to None as it does not support it.
        system_instruction = None

    mode: Union[genai.GenerativeModel, GeminiWrapper]
    model = genai.GenerativeModel(
        model_name=model_name,  # gemini-2.0-flash-exp, gemini-2.0-flash, gemma-3-27b-it
        system_instruction=system_instruction
    )
    if include_wrapper:
        model = GeminiWrapper(model)
    return model
