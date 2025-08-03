from typing import Any, Callable, Iterator, Optional, Union
# noinspection PyPackageRequirements
from haystack.dataclasses import StreamingChunk

# import your concrete classes
# noinspection PyPackageRequirements
from haystack.components.generators import (
    HuggingFaceLocalGenerator,
    HuggingFaceAPIGenerator,
)
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
# (and OllamaGenerator if you have it)


class LLMFacade:
    def __init__(
        self,
        backend: Any,
        *,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        verbose: bool = False
    ):
        """
        backend: an *instance* of any of:
            - HuggingFaceLocalGenerator
            - HuggingFaceAPIGenerator
            - GoogleAIGeminiGenerator
            - LlamaCppGenerator
            - OllamaGenerator
        """
        self._b = backend
        self._stream_cb = streaming_callback
        self.verbose = verbose

        # detect features once
        self.can_stream = hasattr(self._b, "run") and self._is_streaming_impl()
        self.has_context_length = hasattr(self._b, "model") or hasattr(self._b, "pipeline")
        self.has_embedding_dims = hasattr(self._b, "pipeline")  # HF local only

    def _is_streaming_impl(self) -> bool:
        # local HF and API both accept a callback; Gemini streams inherently
        return any(
            hasattr(self._b, attr)
            for attr in ("streaming_callback", "stream")
        )

    @property
    def model_name(self) -> str:
        # many of your classes expose `.model` or `.pipeline.model`
        if hasattr(self._b, "model_name"):
            return self._b.model_name
        if hasattr(self._b, "_model_name"):
            return self._b._model_name
        return self._b.__class__.__name__

    @property
    def context_length(self) -> Optional[int]:
        if not self.has_context_length:
            return None

        # HuggingFaceLocal & API generators wrap an AutoConfig
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(self.model_name)
            return getattr(cfg, "max_position_embeddings", None)
        except Exception:
            return None

    @property
    def embedding_dimensions(self) -> Optional[int]:
        if not self.has_embedding_dims:
            return None
        try:
            # local HF: pipeline.model.config.hidden_size
            return self._b.pipeline.model.config.hidden_size
        except Exception:
            return None

    @property
    def language_model(self) -> Any:
        # yields the actual low-level model object if available
        if hasattr(self._b, "pipeline"):
            return self._b.pipeline.model
        if hasattr(self._b, "model"):
            return getattr(self._b, "model")
        return None

    def generate(self, prompt: str, stream: bool = False) -> Union[str, Iterator[StreamingChunk]]:
        """
        Unified generate API:
          - If stream=True but the backend has no streaming → raises NotImplementedError.
          - Otherwise returns either a plain string or an iterator of StreamingChunk.
        """
        if stream and not self.can_stream:
            raise NotImplementedError(f"{self.model_name} does not support streaming")

        # Many Haystack generators simply expose `.run(prompt)`
        if not stream:
            return self._b.run(prompt)

        # streaming path: backends that accept callbacks will push chunks
        def _iterator():
            # local generators take a `streaming_callback`
            if hasattr(self._b, "streaming_callback"):
                # hijack callback to yield chunks
                def cb(chunk):
                    yield chunk  # not directly possible—see below
                self._b.streaming_callback = cb
                self._b.run(prompt)
            # Gemma/Gemini yields directly
            else:
                for chunk in self._b.run(prompt, stream=True):
                    yield chunk

        return _iterator()

# ---- USAGE EXAMPLE ----


# 1) HF local
hf_local = HuggingFaceLocalGenerator(
    model="gpt2",
    task="text-generation",
    device="cpu",
    streaming_callback=None,
    generation_kwargs={"max_new_tokens": 50}
)
client_local = LLMFacade(hf_local, streaming_callback=lambda c: print(c.text), verbose=True)

# 2) HF API
hf_api = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "gpt2"},
    token="HF_TOKEN",
    streaming_callback=None,
    generation_kwargs={"max_new_tokens": 50},
)
client_api = LLMFacade(hf_api)

# 3) Google Gemini
gem = GoogleAIGeminiGenerator(model="gemini-1.5-flash", api_key="GOOGLE_TOKEN")
client_gemini = LLMFacade(gem, streaming_callback=lambda c: print(c.text))

# 4) Llama-CPP
llama = LlamaCppGenerator(
    model="local-model-path.gguf",
    n_ctx=2048,
    n_batch=512,
    model_kwargs={"n_gpu_layers": -1},
    generation_kwargs={"max_tokens": 50},
)
client_llama = LLMFacade(llama)

# Now your RAGChat can just accept `client: LLMFacade` and always do:
#
#   answer = client.generate("Hello world", stream=False)
#   for chunk in client.generate("Hi", stream=True):
#       handle(chunk)
