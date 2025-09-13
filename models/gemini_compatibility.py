from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Iterable, Iterator, Generic, Any, TypeVar
# noinspection PyPackageRequirements
from google.genai.types import Content, Tool
# noinspection PyPackageRequirements
from google.generativeai.types import (content_types, generation_types, safety_types, GenerateContentResponse,
                                       )

SafetySettingsLike = Dict[str, Any] | safety_types.SafetySettingOptions | None
GenerationConfigLike = Dict[str, Any] | generation_types.GenerationConfigType | None
ToolConfigLike = Dict[str, Any] | List[Tool] | content_types.ToolConfigType | None
SystemInstructionLike = str | content_types.ContentType | None
ToolsLike = Any | content_types.FunctionLibraryType | None
HistoryLike = List[List[str]] | Iterable[Union[str, Dict[str, Any], Content, content_types.StrictContentType]]
GeneratorLike = Union[str, Iterator[str], Iterator[content_types.StrictContentType], GenerateContentResponse]
ContentLike = Union[str, content_types.ContentType]
TResponse = TypeVar("TResponse", bound=GeneratorLike)  # type of send_message’s and generate_content return


# --- Normalizers ---
def normalize_safety_to_dict(safety: Union[Dict[str, Any], safety_types.SafetySettingOptions, None]) -> Dict[str, Any]:
    """
    Normalize a safety setting input to a plain Python dictionary.

    Args:
        safety: The safety settings input. Can be a dict, SafetySettingOptions object, or None.

    Returns:
        A dictionary representation of the safety settings. If `safety` is None, returns an empty dict.
    """
    if safety is None:
        return {}
    if isinstance(safety, dict):
        return safety
    if isinstance(safety, safety_types.SafetySettingOptions):
        return safety_types.to_easy_safety_dict(safety)
    return {"raw": safety}


def normalize_config_to_dict(cfg: Union[Dict[str, Any], generation_types.GenerationConfigType, None]) -> Dict[str, Any]:
    """
    Normalize a generation configuration input to a plain Python dictionary.

    Args:
        cfg: The generation configuration input. Can be a dict, GenerationConfigType object, or None.

    Returns:
        A dictionary representation of the generation configuration. If `cfg` is None, returns an empty dict.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, generation_types.GenerationConfigType):
        return generation_types.to_generation_config_dict(cfg)
    return {"raw": cfg}


def normalize_tool_config_to_dict(cfg: Union[Dict[str, Any], content_types.ToolConfigType, None]) -> Dict[str, Any]:
    """
    Normalize a tool configuration input to a plain Python dictionary.

    Args:
        cfg: The tool configuration input. Can be a dict, ToolConfigType object, or None.

    Returns:
        A dictionary representation of the tool configuration. If `cfg` is None, returns an empty dict.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, content_types.ToolConfigType):
        return cfg.to_dict()
    return {"raw": cfg}


def normalize_instruction_to_str(instr: Union[str, content_types.ContentType, None]) -> str:
    """
    Normalize a system instruction input to a string.

    Args:
        instr: The system instruction. Can be a string, ContentType object, or None.

    Returns:
        The instruction text as a string. Returns an empty string if `instr` is None.
    """
    if instr is None:
        return ""
    if isinstance(instr, str):
        return instr
    if isinstance(instr, content_types.ContentType):
        return getattr(instr, "text", str(instr))
    return str(instr)


class GeminiChatSessionCompatible(ABC, Generic[TResponse]):
    """
    Abstract base class representing a Gemini-compatible chat session.

    This class mimics the interface of the Google SDK `start_chat` object
    and provides a uniform API for sending messages and retrieving chat history.
    """

    def __init__(self, history: Optional[HistoryLike]) -> None:
        """
        Initialize the chat session wrapper.

        Args:
            history: Optional initial chat history in Gemini-like format.
        """
        self._history: HistoryLike = history or []

    @abstractmethod
    def send_message(self, contents: ContentLike, **kwargs) -> TResponse:
        """
        Send a message in the ongoing chat.

        Args:
            contents: The content to send. Can be a string or ContentType object.
            **kwargs: Additional provider-specific arguments.

        Returns:
            A response object of type TResponse, which may be a string, iterator, or other generator.
        """
        pass

    def get_history(self) -> HistoryLike:
        """
        Return the full chat history in Gemini-like format.

        Returns:
            The chat history as a HistoryLike object.
        """
        return self._history


# This class is not strictly necessary. LLMModel can use Gemini directly.
# However, it is useful to have a minimal Gemini-like interface that other providers can implement.
# This allows for easier switching between providers if needed.
class GeminiCompatible(ABC, Generic[TResponse]):
    """
    Abstract base class representing a Gemini-compatible model interface.

    Subclasses should implement content generation and chat session creation,
    accepting either Gemini SDK types or standard Python types for input.
    """

    def __init__(
        self,
        model_or_name: Any,
        safety_settings: SafetySettingsLike = None,
        generation_config: GenerationConfigLike = None,
        tools: ToolsLike = None,
        tool_config: ToolConfigLike = None,
        system_instruction: SystemInstructionLike = None,
        secret_token: str | None = None,
        normalize: bool = False,
    ) -> None:
        """
        Initialize the Gemini-compatible model wrapper.

        Args:
            model_or_name: Either a model instance or a string model name.
            safety_settings: Optional safety configuration settings.
            generation_config: Optional generation configuration settings.
            tools: Optional function library or tools object.
            tool_config: Optional tool configuration.
            system_instruction: Optional system instruction for the model.
            secret_token: Optional API secret token.
            normalize: If True, normalize all input types to standard dicts/strings.
        """
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

        self._safety_settings = normalize_safety_to_dict(safety_settings) if normalize else safety_settings
        self._generation_config = normalize_config_to_dict(generation_config) if normalize else generation_config
        self._tool_config = normalize_tool_config_to_dict(tool_config) if normalize else tool_config
        self._system_instruction = normalize_instruction_to_str(system_instruction) if normalize else system_instruction

    @property
    def model_name(self) -> str:
        """
        Get the model's name.

        Returns:
            The name of the model.
        """
        return self._model_name

    @property
    def system_instruction(self) -> SystemInstructionLike:
        """
        Get the system instruction.

        Returns:
            The system instruction as a string or ContentType.
        """
        return self._system_instruction

    @property
    def generation_config(self) -> GenerationConfigLike:
        """
        Get the generation configuration.

        Returns:
            The model's generation configuration.
        """
        return self._generation_config

    @property
    def safety_settings(self) -> SafetySettingsLike:
        """
        Get the safety settings.

        Returns:
            The model's safety settings.
        """
        return self._safety_settings

    @property
    def tool_config(self) -> ToolConfigLike:
        """
        Get the tool configuration.

        Returns:
            The model's tool configuration.
        """
        return self._tool_config

    @property
    def tools(self) -> ToolsLike:
        """
        Get the function library or tools object.

        Returns:
            The model's tools object.
        """
        return self._tools

    @abstractmethod
    def generate_content(self, contents: ContentLike, **kwargs) -> TResponse:
        """
        Generate text or content for a single prompt.

        Args:
            contents: The content to generate text for.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The generated content as a TResponse object.
        """
        pass

    @abstractmethod
    def start_chat(self, history: HistoryLike) -> GeminiChatSessionCompatible[TResponse]:
        """
        Start a chat session with optional history.

        Args:
            history: Chat history to initialize the session with.

        Returns:
            An object implementing GeminiChatSessionCompatible, representing the chat session.
        """
        pass

    # --- transparent forwarding ---
    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying model if not found in this wrapper.

        Args:
            name: The attribute name to access.

        Returns:
            The value of the attribute from the underlying model.
        """
        return getattr(self._model, name)
