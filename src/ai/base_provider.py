# src/agents/ai/provider.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum


class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    # Add more providers here


class AIMessage:
    def __init__(self, role: str, content: str, name: Optional[str] = None, **kwargs):
        self.role = role
        self.content = content
        self.name = name
        self.additional_params = kwargs

    def to_dict(self) -> Dict[str, Any]:
        message_dict = {"role": self.role, "content": self.content}
        if self.name:
            message_dict["name"] = self.name
        message_dict.update(self.additional_params)
        return message_dict


class AIResponse:
    def __init__(
        self, content: str, raw_response: Any, usage: Dict[str, int], model: str
    ):
        self.content = content
        self.raw_response = raw_response
        self.usage = usage
        self.model = model


class BaseAIProvider(ABC):
    def __init__(self, api_key: str, **kwargs):
        self.api_key = api_key
        self.additional_config = kwargs

    @abstractmethod
    async def generate_text(
        self, provider_type: AIProvider, messages: List[AIMessage], model: str, **kwargs
    ) -> AIResponse:
        """Generate text using the specified provider"""
        raise NotImplementedError("generate_text method not implemented")

    @abstractmethod
    async def generate_speech(
        self,
        provider_type: AIProvider,
        text: str,
        model: str | None = None,
        name: str | None = None,
        speed: float | None = None,
    ) -> Any:
        """Generate speech using the specified provider"""
        # I think this is not needed for the current implementation
        pass


class AIProviderRepository(ABC):
    """Interface for AI provider repositories"""

    def __init__(self):
        self._providers: Dict[AIProvider, BaseAIProvider] = {}

    def register_provider(self, provider_type: AIProvider, provider: BaseAIProvider):
        """Register a new AI provider"""
        self._providers[provider_type] = provider

    def get_provider(self, provider_type: AIProvider) -> BaseAIProvider:
        """Get an AI provider by type"""
        if provider_type not in self._providers:
            raise KeyError(f"Provider {provider_type} not registered")
        return self._providers[provider_type]

    def is_empty(self) -> bool:
        """Check if there are any registered providers"""
        return not bool(self._providers)

    async def generate_text(
        self, provider_type: AIProvider, messages: List[AIMessage], model: str, **kwargs
    ) -> AIResponse:
        """Generate text using the specified provider"""
        provider = self.get_provider(provider_type)
        return await provider.generate_text(provider_type, messages, model, **kwargs)

    async def generate_speech(
        self,
        provider_type: AIProvider,
        text: str,
        model: str | None = None,
        name: str | None = None,
        speed: float | None = None,
    ) -> Any:
        """Generate speech using the specified provider"""
        provider = self.get_provider(provider_type)
        return await provider.generate_speech(provider_type, text, model, name, speed)
