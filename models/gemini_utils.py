# noinspection PyPackageRequirements
import google.generativeai as genai
from google.auth.exceptions import DefaultCredentialsError
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union


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
        valid_models = []
        for model in genai.list_models():
            # Remove "model/" prefix if present
            model_name = model.name
            if model_name.startswith("models/"):
                model_name = model.name[len("models/"):]
            valid_models.append(model_name)
        return valid_models
    except DefaultCredentialsError:
        return VALID_GEMINI_MODELS


# Taken from https://medium.com/latinxinai/simple-chatbot-gradio-google-gemini-api-4ce02fbaf09f
def chat_to_gemini_format(history: List[List[str]]) -> List[Dict[str, Any]]:
    new_history = []
    for chat_response in history:
        new_history.append({"parts": [{"text": chat_response[0]}], "role": "user"})
        new_history.append({"parts": [{"text": chat_response[1]}], "role": "model"})
    return new_history


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


# This class is not strictly necessary. LLMModel can use Gemini directly.
# However, it is useful to have a minimal Gemini-like interface that other providers can implement.
# This allows for easier switching between providers if needed.
class MinGeminiCompatible(ABC):
    """
    Abstract Gemini-like interface that other providers must implement.
    """

    def __init__(self, model_name: str, system_instruction: Optional[str] = None, secret_token: Optional[str] = None):
        self._model_name = model_name
        self._system_instruction = system_instruction
        self._secret_token = secret_token

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system_instruction(self) -> Optional[str]:
        return self._system_instruction

    @abstractmethod
    def generate_content(self, prompt: str, **kwargs) -> str:
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

    def __init__(self, history: Union[List[List[str]], Optional[List[Dict[str, Any]]]] = None):
        self._history: Union[List[List[str]], Optional[List[Dict[str, Any]]]] = history or []

    @abstractmethod
    def send_message(self, prompt: str, **kwargs) -> str:
        """
        Send a message in the ongoing chat and return the model's reply.
        """
        pass

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the full chat history in Gemini-like format."""
        return self._history

