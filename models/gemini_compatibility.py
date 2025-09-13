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
    if safety is None:
        return {}
    if isinstance(safety, dict):
        return safety
    if isinstance(safety, safety_types.SafetySettingOptions):
        return safety_types.to_easy_safety_dict(safety)
    return {"raw": safety}


def normalize_config_to_dict(cfg: Union[Dict[str, Any], generation_types.GenerationConfigType, None]) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, generation_types.GenerationConfigType):
        return generation_types.to_generation_config_dict(cfg)
    return {"raw": cfg}


def normalize_tool_config_to_dict(cfg: Union[Dict[str, Any], content_types.ToolConfigType, None]) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, content_types.ToolConfigType):
        return cfg.to_dict()
    return {"raw": cfg}


def normalize_instruction_to_str(instr: Union[str, content_types.ContentType, None]) -> str:
    if instr is None:
        return ""
    if isinstance(instr, str):
        return instr
    if isinstance(instr, content_types.ContentType):
        return getattr(instr, "text", str(instr))
    return str(instr)


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


# This class is not strictly necessary. LLMModel can use Gemini directly.
# However, it is useful to have a minimal Gemini-like interface that other providers can implement.
# This allows for easier switching between providers if needed.
class MinGeminiCompatible(ABC, Generic[TResponse]):
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

        self._safety_settings = normalize_safety_to_dict(safety_settings) if normalize else safety_settings
        self._generation_config = normalize_config_to_dict(generation_config) if normalize else generation_config
        self._tool_config = normalize_tool_config_to_dict(tool_config) if normalize else tool_config
        self._system_instruction = normalize_instruction_to_str(system_instruction) if normalize else system_instruction

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
    def generate_content(self, contents: ContentLike, **kwargs) -> TResponse:
        """
        Generate text for a single prompt.
        """
        pass

    @abstractmethod
    def start_chat(self, history: HistoryLike) -> GeminiChatSessionCompatible[TResponse]:
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
