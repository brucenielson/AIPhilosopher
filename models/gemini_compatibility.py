from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Callable, Iterable, Iterator
import re
import time
# noinspection PyPackageRequirements
from google.api_core.exceptions import ResourceExhausted
# noinspection PyPackageRequirements
from google.genai.types import Content, Tool
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
from typing import TypeVar, Any

SafetySettingsLike = Dict[str, Any] | safety_types.SafetySettingOptions | None
GenerationConfigLike = Dict[str, Any] | generation_types.GenerationConfigType | None
ToolConfigLike = Dict[str, Any] | List[Tool] | content_types.ToolConfigType | None
SystemInstructionLike = str | content_types.ContentType | None
ToolsLike = Any | content_types.FunctionLibraryType | None
HistoryLike = List[List[str]] | List[Dict[str, Any]] | List[Content] | None
GeneratorLike = str | Iterator[str] | GenerateContentResponse
TResp = TypeVar("TResp")  # type of send_message’s return
TChat = TypeVar("TChat", bound="GeminiChatSessionCompatible[Any]")  # type of chat session


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


# --- Normalizers ---
def normalize_safety(safety: Union[Dict[str, Any], safety_types.SafetySettingOptions, None]) -> Dict[str, Any]:
    if safety is None:
        return {}
    if isinstance(safety, dict):
        return safety
    if isinstance(safety, safety_types.SafetySettingOptions):
        return safety_types.to_easy_safety_dict(safety)
    return {"raw": safety}


def normalize_config(cfg: Union[Dict[str, Any], generation_types.GenerationConfigType, None]) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, generation_types.GenerationConfigType):
        return generation_types.to_generation_config_dict(cfg)
    return {"raw": cfg}


def normalize_tool_config(cfg: Union[Dict[str, Any], content_types.ToolConfigType, None]) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, content_types.ToolConfigType):
        return cfg.to_dict()
    return {"raw": cfg}


def normalize_instruction(instr: Union[str, content_types.ContentType, None]) -> str:
    if instr is None:
        return ""
    if isinstance(instr, str):
        return instr
    if isinstance(instr, content_types.ContentType):
        return getattr(instr, "text", str(instr))
    return str(instr)


class GeminiChatSessionCompatible(ABC):
    """
    Gemini-compatible chat session interface.
    Mimics the Google SDK `start_chat` object.
    """

    def __init__(self, history: HistoryLike) -> None:
        self._history: HistoryLike = history or []

    @abstractmethod
    def send_message(self, contents: str, **kwargs) -> GeneratorLike:
        """
        Send a message in the ongoing chat and return the model's reply.
        """
        pass

    def get_history(self) -> HistoryLike:
        """Return the full chat history in Gemini-like format."""
        return self._history


class GeminiChatSessionWrapper(GeminiChatSessionCompatible):
    def __init__(self, chat_session: ChatSession, history: HistoryLike = None) -> None:
        super().__init__(history=history)
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

    def get_history(self) -> HistoryLike:
        # override to return SDK-tracked history
        return list(self._chat_session.history)

    def __getattr__(self, name: str):
        # delegate any missing methods to the underlying Gemini chat object
        return getattr(self._chat_session, name)


# This class is not strictly necessary. LLMModel can use Gemini directly.
# However, it is useful to have a minimal Gemini-like interface that other providers can implement.
# This allows for easier switching between providers if needed.
class MinGeminiCompatible(ABC):
    """
    Abstract Gemini-like interface that other providers must implement.
    Accepts Gemini SDK types or plain dicts/strings, normalizes them internally.
    """
    def __init__(
        self,
        model_or_name: Any | str,
        safety_settings: SafetySettingsLike = None,
        generation_config: GenerationConfigLike = None,
        tools: ToolsLike = None,
        tool_config: ToolConfigLike = None,
        system_instruction: SystemInstructionLike = None,
        secret_token: str | None = None,
        normalize: bool = False,
    ) -> None:
        self._model: Any | None = None
        if isinstance(model_or_name, str):
            self._model_name = model_or_name
        else:
            self._model_name = getattr(model_or_name, "model_name", "anonymous-model")
            self._model = model_or_name

        self._secret_token: str | None = secret_token
        self._tools: ToolsLike = tools

        # normalize Gemini types → neutral Python dicts/strings
        self._safety_settings: SafetySettingsLike
        self._generation_config: GenerationConfigLike
        self._tool_config: ToolConfigLike
        self._system_instruction: SystemInstructionLike

        self._safety_settings = normalize_safety(safety_settings) if normalize else safety_settings
        self._generation_config = normalize_config(generation_config) if normalize else generation_config
        self._tool_config = normalize_tool_config(tool_config) if normalize else tool_config
        self._system_instruction = normalize_instruction(system_instruction) if normalize else system_instruction

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system_instruction(self) -> SystemInstructionLike:
        return self._system_instruction

    @property
    def generation_config(self) -> GenerationConfigLike:
        return self._generation_config

    @property
    def safety_settings(self) -> SafetySettingsLike:
        return self._safety_settings

    @property
    def tool_config(self) -> ToolConfigLike:
        return self._tool_config

    @property
    def tools(self) -> ToolsLike:
        return self._tools

    @abstractmethod
    def generate_content(self, contents: str, **kwargs) -> GeneratorLike:
        """
        Generate text for a single prompt.
        """
        pass

    @abstractmethod
    def start_chat(self, history: List[Dict[str, Any]]) -> GeminiChatSessionCompatible:
        """
        Start a chat session with an optional history.
        Should return an object that has .send_message(prompt) -> str
        """
        pass

    # --- transparent forwarding ---
    def __getattr__(self, name: str):
        """
        If the attribute is not found on this wrapper,
        delegate it to the underlying GenerativeModel.
        """
        return getattr(self._model, name)


class GeminiWrapper(MinGeminiCompatible):
    """
    Gemini wrapper that is compatible with MinGeminiCompatible but still
    exposes the full underlying GenerativeModel API transparently.
    """

    def __init__(
        self,
        model_or_name: Any | str,
        safety_settings: SafetySettingsLike = None,
        generation_config: GenerationConfigLike = None,
        tools: ToolsLike = None,
        tool_config: ToolConfigLike = None,
        system_instruction: SystemInstructionLike = None,
        secret_token: str | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            model_or_name=model_or_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
            secret_token=secret_token,
            normalize=normalize,
        )

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
            history: Iterable[content_types.StrictContentType] | None = None,
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
        return GeminiChatSessionWrapper(chat)

    @property
    def model_name(self) -> str:
        return self._model.model_name
