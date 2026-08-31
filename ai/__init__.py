"""
Movie Review Engine - AI Module
"""

from typing import Optional
from .base import (
    BaseAIProvider,
    VisualAnalysis,
    EventUncertainty,
    EventRecord,
    EventsDocument,
    AIProviderError,
    MissingAPIKeyError,
    ModelInvocationError,
    InvalidModelResponseError,
)
from .providers.gemini import GeminiAIProvider
from .providers.mock import MockAIProvider


def get_ai_provider(
    provider_name: str = "gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseAIProvider:
    """
    Factory function to instantiate the requested multimodal AI provider.
    """
    p_lower = provider_name.strip().lower()
    if p_lower == "gemini":
        return GeminiAIProvider(model_name=model_name, api_key=api_key)
    elif p_lower in ("mock", "fake", "test"):
        return MockAIProvider(model_name=model_name or "mock-vision-v1", api_key=api_key)
    else:
        raise ValueError(f"Unsupported AI provider '{provider_name}'. Supported: 'gemini', 'mock'.")


__all__ = [
    "BaseAIProvider",
    "VisualAnalysis",
    "EventUncertainty",
    "EventRecord",
    "EventsDocument",
    "AIProviderError",
    "MissingAPIKeyError",
    "ModelInvocationError",
    "InvalidModelResponseError",
    "GeminiAIProvider",
    "MockAIProvider",
    "get_ai_provider",
]
