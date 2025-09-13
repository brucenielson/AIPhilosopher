# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.auth.exceptions import DefaultCredentialsError
from typing import Optional, List, Dict, Any, Union
from models.gemini_wrapper import GeminiWrapper

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

# module-level cache
_gemini_models_cache: List[str] | None = None


def get_gemini_models(secret_token: Optional[str] = None, use_cache: bool = True) -> List[str]:
    global _gemini_models_cache

    if use_cache and _gemini_models_cache is not None:
        return _gemini_models_cache
    try:
        if secret_token:
            genai.configure(api_key=secret_token)

        model_list: List[str] = [
            name
            for model in genai.list_models()
            for name in (model.name, model.name.removeprefix("models/"))
        ]
        _gemini_models_cache = model_list
        return model_list
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
