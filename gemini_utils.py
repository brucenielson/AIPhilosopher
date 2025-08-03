# noinspection PyPackageRequirements
from google.api_core.exceptions import ResourceExhausted
# noinspection PyPackageRequirements
from google.generativeai import ChatSession
# noinspection PyPackageRequirements
from google.generativeai.types.generation_types import GenerationConfig, GenerateContentResponse
# noinspection PyPackageRequirements
import google.generativeai as genai
# noinspection PyPackageRequirements
from google.generativeai.types import Tool
from typing import Any, List, Union, Optional
import time
import re


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


def _extract_retry_seconds(exc: ResourceExhausted, default: int = 15) -> int:
    """
    Extracts retry_delay.seconds from the exception's details text.
    Falls back to `default` if not found or parsing fails.
    """
    try:
        details = str(getattr(exc, "details", ""))
        match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', details)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return default


def send_message(model: Union[ChatSession, genai.GenerativeModel],
                 message: str,
                 tools: List[Tool] = None,
                 stream: bool = False,
                 config: GenerationConfig = None,
                 **generation_kwargs: Any) -> Union[GenerateContentResponse, str]:

    if config is None and generation_kwargs:
        # Set up the config with any provided generation parameters
        config: GenerationConfig = GenerationConfig(**generation_kwargs)

    try:
        if isinstance(model, ChatSession):
            # If tools are provided, include them in the message
            return model.send_message(message,
                                      generation_config=config,
                                      tools=tools,
                                      stream=stream)
        else:
            # Otherwise, generate content directly without a chat session
            response = model.generate_content(
                contents=message,
                generation_config=config,
                tools=tools,
                stream=stream
            )
            return getattr(response, "text", None) or "[No response text]"

    except ResourceExhausted as e:
        # Handle rate limit errors by checking for retry_delay
        delay = _extract_retry_seconds(e)
        if delay is None or delay <= 0:
            delay = 15
        print(f"\nRate limit exceeded. Retrying in {delay} seconds...")
        time.sleep(delay)
        return send_message(model, message, tools=tools, stream=stream, config=config, **generation_kwargs)

    except Exception as e:
        print(f"Error during chat message sending: {e}")
        raise
