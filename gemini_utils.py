# noinspection PyPackageRequirements
import google.generativeai as genai
from typing import Optional

# List of valid Gemini model_or_name names.
VALID_GEMINI_MODELS = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemma-3-27b-it",
    "gemma-3-8b-it",
    "gemma-3-8b-it-v1",
    "gemma-3-8b-it-v2"
]


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
