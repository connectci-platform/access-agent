"""LLM provider abstraction supporting OpenAI, vLLM, and custom endpoints."""

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config import settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_chat_model(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> BaseChatModel:
        """Get a chat model instance.

        Args:
            model_name: Override the default model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            A LangChain chat model instance.
        """


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API."""

    def __init__(self, api_key: str, default_model: str = "gpt-4o"):
        self.api_key = SecretStr(api_key)
        self.default_model = default_model

    def get_chat_model(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> BaseChatModel:
        return ChatOpenAI(
            model=model_name or self.default_model,
            api_key=self.api_key,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible APIs (vLLM, LocalAI, etc.).

    This works with any server that implements the OpenAI chat completions API,
    including:
    - vLLM with --served-model-name
    - LocalAI
    - Ollama with OpenAI compatibility
    - Any custom endpoint
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy",  # Many local servers don't require a key
        default_model: str = "default",
    ):
        self.base_url = base_url
        self.api_key = SecretStr(api_key)
        self.default_model = default_model

    def get_chat_model(
        self,
        model_name: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> BaseChatModel:
        return ChatOpenAI(
            model=model_name or self.default_model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )


def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider.

    Returns the appropriate provider based on LLM_PROVIDER setting:
    - openai: Uses OpenAI API
    - vllm: Uses vLLM server with OpenAI-compatible API
    - access_ai: Uses ACCESS AI endpoint

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If the provider is unknown or misconfigured.
    """
    provider = settings.LLM_PROVIDER

    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for openai provider")
        return OpenAIProvider(
            api_key=settings.OPENAI_API_KEY,
            default_model=settings.OPENAI_MODEL,
        )

    if provider == "vllm":
        return OpenAICompatibleProvider(
            base_url=settings.VLLM_BASE_URL,
            api_key=settings.VLLM_API_KEY or "dummy",
            default_model=settings.VLLM_MODEL_NAME,
        )

    if provider == "access_ai":
        if not settings.ACCESS_AI_API_KEY:
            raise ValueError("ACCESS_AI_API_KEY is required for access_ai provider")
        return OpenAICompatibleProvider(
            base_url=settings.ACCESS_AI_BASE_URL,
            api_key=settings.ACCESS_AI_API_KEY,
            default_model="access-llama",
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm(
    model_name: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
) -> BaseChatModel:
    """Get a configured LLM instance.

    Convenience function that gets the provider and returns a model.

    Args:
        model_name: Override the default model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.

    Returns:
        A LangChain chat model instance.
    """
    provider = get_llm_provider()
    return provider.get_chat_model(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
