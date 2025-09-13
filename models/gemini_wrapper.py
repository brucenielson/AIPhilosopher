from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Callable, Iterable, Iterator, Generic, Any, TypeVar, ParamSpec
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
from models.gemini_compatibility import MinGeminiCompatible
import functools
from utilities.general_utils import logger


SafetySettingsLike = Dict[str, Any] | safety_types.SafetySettingOptions | None
GenerationConfigLike = Dict[str, Any] | generation_types.GenerationConfigType | None
ToolConfigLike = Dict[str, Any] | List[Tool] | content_types.ToolConfigType | None
SystemInstructionLike = str | content_types.ContentType | None
ToolsLike = Any | content_types.FunctionLibraryType | None
HistoryLike = List[List[str]] | Iterable[Union[str, Dict[str, Any], Content, content_types.StrictContentType]]
GeneratorLike = Union[str, Iterator[str], Iterator[content_types.StrictContentType], GenerateContentResponse]
ContentLike = Union[str, content_types.ContentType]
TResponse = TypeVar("TResponse", bound=GeneratorLike)  # type of send_message’s and generate_content return


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
            logger.warning(f"\nGemini Rate limit exceeded. Retrying in {delay} seconds... (attempt {attempts+1})")
            time.sleep(delay)

        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                delay = 15
                logger.warning(f"Rate limit-like error detected from provider. "
                               f"Retrying in {delay} seconds... (attempt {attempts+1})")
                time.sleep(delay)
            else:
                logger.error(f"Error during chat message sending: {e}")
                raise

        attempts += 1

    raise RuntimeError(f"Max retries exceeded ({max_retries}) for {fn.__name__}")


P = ParamSpec("P")
R = TypeVar("R")


def retryable(max_retries: int = 5) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return with_retry(fn, *args, max_retries=max_retries, **kwargs)
        return wrapper
    return decorator


# --- Normalizers ---
def normalize_history_to_gemini_content(
    history: Optional[HistoryLike]
) -> List[content_types.StrictContentType]:
    """
    Convert a heterogeneous history input to a flat list of StrictContentType objects.

    Supports:
    - List[List[str]]  → each inner list [user, model] becomes two items
    - Iterable[Union[str, dict, Content, StrictContentType]]
    """
    if history is None:
        return []

    normalized: List[content_types.StrictContentType] = []

    # Handle the special List[List[str]] case
    if isinstance(history, list) and history and all(
        isinstance(item, list) and len(item) == 2 for item in history
    ):
        for user_text, model_text in history:
            normalized.append(content_types.to_content(user_text))
            normalized.append(content_types.to_content(model_text))
        return normalized

    # Otherwise handle generic iterable
    for item in history:
        if isinstance(item, content_types.StrictContentType):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(content_types.to_content(item))
        elif isinstance(item, Content):
            normalized.append(item)
        elif isinstance(item, dict):
            normalized.append(content_types.to_content(item))
        else:
            raise TypeError(f"Unsupported history item type: {type(item)}")

    return normalized


class GeminiChatSessionCompatible(ABC, Generic[TResponse]):
    """
    Gemini-compatible chat session interface.
    Mimics the Google SDK `start_chat` object.
    """

    def __init__(self, history: Optional[HistoryLike]) -> None:
        self._history: HistoryLike = history or []

    @abstractmethod
    def send_message(self, contents: ContentLike, **kwargs) -> TResponse:
        """
        Send a message in the ongoing chat and return the model's reply.
        """
        pass

    def get_history(self) -> HistoryLike:
        """Return the full chat history in Gemini-like format."""
        return self._history


class GeminiChatSessionWrapper(GeminiChatSessionCompatible[TResponse], Generic[TResponse]):
    def __init__(self, chat_session: ChatSession, history: Optional[HistoryLike] = None) -> None:
        super().__init__(history=history)
        self._chat_session: ChatSession = chat_session

    @retryable(max_retries=5)
    def send_message(
            self,
            content: ContentLike,
            *,
            generation_config: Optional[generation_types.GenerationConfigType] = None,
            safety_settings: Optional[safety_types.SafetySettingOptions] = None,
            stream: bool = False,
            tools: Optional[content_types.FunctionLibraryType] = None,
            tool_config: Optional[content_types.ToolConfigType] = None,
            request_options: Optional[helper_types.RequestOptionsType] = None,
    ) -> TResponse:
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


class GeminiWrapper(MinGeminiCompatible[GenerateContentResponse]):
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
    ) -> TResponse:
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
            history: Optional[HistoryLike] = None,
            enable_automatic_function_calling: bool = False,
    ) -> GeminiChatSessionWrapper[TResponse]:
        """
        Wrapper around Gemini's start_chat.
        Mirrors the official signature.
        """
        history = normalize_history_to_gemini_content(history)
        chat = self._model.start_chat(
            history=history,
            enable_automatic_function_calling=enable_automatic_function_calling,
        )
        return GeminiChatSessionWrapper[TResponse](chat)

    @property
    def model_name(self) -> str:
        return self._model.model_name
