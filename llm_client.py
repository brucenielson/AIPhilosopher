# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.generativeai import ChatSession
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig, GenerateContentResponse
# noinspection PyPackageRequirements
from google.generativeai.types import Tool
from typing import Any, List, Union, Optional, Dict
from gemini_utils import send_message


class LLMClient:
    def __init__(self, model: Union[genai.GenerativeModel],
                 *,
                 password: Optional[str] = None,
                 system_instruction: Optional[str] = None,
                 tools: List[Tool] = None,
                 config: GenerationConfig = None,
                 **generation_kwargs: Any
                 ):

        self._model: Union[genai.GenerativeModel] = model
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

    def send_message(self,
                     message: str,
                     chat_history: Optional[List[Dict[str, Any]]] = None,
                     stream: bool = False,
                     tools: List[Tool] = None,
                     config: GenerationConfig = None,
                     **generation_kwargs: Any
) -> Union[GenerateContentResponse, str]:

        model: Union[genai.GenerativeModel, ChatSession]
        if chat_history is None:
            model = self._model
        else:
            model = self._model.start_chat(history=chat_history)

        return send_message(model,
                            message,
                            tools=tools if tools is not None else self._tools,
                            stream=stream,
                            config=config if config is not None else self._config,
                            **generation_kwargs)
