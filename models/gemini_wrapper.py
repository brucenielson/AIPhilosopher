from typing import Optional, List, Callable, Generic, Any, TypeVar, ParamSpec
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
from models.gemini_compatibility import (
    GeminiCompatible,
    SafetySettingsLike,
    GenerationConfigLike,
    ToolConfigLike,
    SystemInstructionLike,
    ToolsLike,
    HistoryLike,
    ContentLike,
    TResponse,
    GeminiChatSessionCompatible,
)
import functools
from utilities.general_utils import logger


def gemini_extract_retry_seconds(exc: ResourceExhausted, default: int = 15) -> int:
    """
    Extract the suggested retry delay in seconds from a ResourceExhausted exception.

    Args:
        exc: The ResourceExhausted exception instance.
        default: Default number of seconds to retry if parsing fails.

    Returns:
        The number of seconds to wait before retrying the request.
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
    Call a function with retry logic for rate-limit errors from Gemini or similar APIs.

    Handles:
      - ResourceExhausted exceptions with optional retry_delay.
      - Generic rate-limit errors containing 'rate limit' or HTTP 429.

    Args:
        fn: The function to call.
        *args: Positional arguments for the function.
        max_retries: Maximum number of retries before failing.
        **kwargs: Keyword arguments for the function.

    Returns:
        The return value of the function.

    Raises:
        RuntimeError: If the maximum number of retries is exceeded.
        Exception: Any other exceptions raised by `fn` not matching rate-limit errors.
    """
    attempts = 0
    while attempts <= max_retries:
        try:
            return fn(*args, **kwargs)
        except ResourceExhausted as e:
            delay = gemini_extract_retry_seconds(e) or 15
            logger.warning(f"[Retry {attempts + 1}/{max_retries}] Gemini rate limit exceeded, retrying in {delay}s")
            time.sleep(delay)
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                delay = 15
                logger.warning(f"[Retry {attempts + 1}/{max_retries}] "
                               f"Rate-limit-like error detected, retrying in {delay}s")
                time.sleep(delay)
            else:
                logger.error(f"Error during chat message sending: {e}")
                raise
        attempts += 1
    raise RuntimeError(f"Max retries exceeded ({max_retries}) for {fn.__name__}")


P = ParamSpec("P")
R = TypeVar("R")


def retryable(max_retries: int = 5) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to apply retry logic to a function using `with_retry`.

    Args:
        max_retries: Maximum number of retry attempts.

    Returns:
        A decorator function.
    """
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
    Convert a heterogeneous history input into a flat list of StrictContentType objects.

    Supports:
        - List[List[str]]: each inner list [user, model] becomes two StrictContentType items.
        - Iterable[Union[str, dict, Content, StrictContentType]]: converts each element individually.

    Args:
        history: Optional chat history in a variety of formats.

    Returns:
        A list of StrictContentType objects suitable for Gemini SDK.

    Raises:
        ValueError: If an inner list in list-of-lists format does not have exactly 2 elements.
        TypeError: If an unsupported item type is found in the history iterable.
    """
    if history is None:
        return []

    normalized: List[content_types.StrictContentType] = []

    # Handle the special List[List[str]] case
    if isinstance(history, list) and history and all(isinstance(item, list) for item in history):
        for idx, item in enumerate(history):
            if len(item) != 2:
                raise ValueError(f"Inner history list at index {idx} must have exactly 2 elements: [user, model]")
            user_text, model_text = item
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


class GeminiChatSessionWrapper(GeminiChatSessionCompatible[TResponse], Generic[TResponse]):
    """
    Wrapper around Gemini ChatSession to make it compatible with GeminiChatSessionCompatible interface.

    Adds:
      - Retry logic for sending messages.
      - Conversion of string messages into SDK Content objects.
      - Transparent access to underlying ChatSession methods.
    """

    def __init__(self, chat_session: ChatSession, history: Optional[HistoryLike] = None) -> None:
        """
        Initialize a GeminiChatSessionWrapper.

        Args:
            chat_session: The underlying Gemini ChatSession instance.
            history: Optional initial chat history.
        """
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
        Send a message through the wrapped ChatSession with retry handling.

        Args:
            content: The message content (string or ContentType object).
            generation_config: Optional generation configuration.
            safety_settings: Optional safety settings.
            stream: Whether to stream the response.
            tools: Optional tools library.
            tool_config: Optional tool configuration.
            request_options: Optional request options.

        Returns:
            The response from the ChatSession.

        Raises:
            Exception: Any exception not related to rate limits will propagate.
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
        """
        Return the chat session's current history.

        Returns:
            The history list from the underlying ChatSession.
        """
        return list(self._chat_session.history)

    def __getattr__(self, name: str):
        """
        Delegate missing attributes to the underlying ChatSession.

        Args:
            name: Attribute name.

        Returns:
            The attribute from the underlying ChatSession.
        """
        return getattr(self._chat_session, name)


class GeminiWrapper(GeminiCompatible[GenerateContentResponse]):
    """
    High-level wrapper around Gemini generative models providing:
      - Compatibility with GeminiCompatible interface.
      - Transparent access to underlying model methods.
      - Retry logic on rate-limit errors for generate_content and start_chat.
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
        """
        Initialize the GeminiWrapper.

        Args:
            model_or_name: Either a model object or a string model name.
            safety_settings: Optional safety settings.
            generation_config: Optional generation configuration.
            tools: Optional tools library.
            tool_config: Optional tool configuration.
            system_instruction: Optional system instruction string/content.
            secret_token: Optional API secret token.
            normalize: Whether to normalize Gemini types to dict/string.
        """
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
        """
        Generate content from the underlying Gemini model with retry support.

        Args:
            contents: The input prompt(s).
            generation_config: Optional generation configuration.
            safety_settings: Optional safety settings.
            stream: Whether to stream the output.
            tools: Optional tools library.
            tool_config: Optional tool configuration.
            request_options: Optional request options.

        Returns:
            The generated response.

        Raises:
            RuntimeError: If the underlying model instance is not set.
        """
        if self._model is None:
            raise RuntimeError("Underlying model instance is not set. Provide a model object, not a string.")
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
        Start a chat session with the underlying Gemini model.

        Args:
            history: Optional chat history.
            enable_automatic_function_calling: Whether to allow automatic function calling.

        Returns:
            A GeminiChatSessionWrapper instance.

        Raises:
            RuntimeError: If the underlying model instance is not set.
        """
        if self._model is None:
            raise RuntimeError("Underlying model instance is not set. Provide a model object, not a string.")

        history = normalize_history_to_gemini_content(history)
        chat = self._model.start_chat(
            history=history,
            enable_automatic_function_calling=enable_automatic_function_calling,
        )
        return GeminiChatSessionWrapper[TResponse](chat)

    @property
    def model_name(self) -> str:
        """
        Return the name of the underlying Gemini model.

        Returns:
            Model name as a string.

        Raises:
            RuntimeError: If the underlying model instance is not set.
        """
        if self._model is None:
            raise RuntimeError("Underlying model instance is not set. Provide a model object, not a string.")
        return self._model.model_name
