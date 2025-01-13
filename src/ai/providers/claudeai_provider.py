from typing import Any, List, Optional
from src.agents.ai.base_provider import (
    AIMessage,
    AIResponse,
    BaseAIProvider,
)


class ClaudeAIProvider(BaseAIProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)

    async def generate_text(
        self,
        messages: List[AIMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AIResponse:
        raise NotImplementedError("generate_text method not implemented")

    async def generate_speech(
        self,
        text: str,
        model: str | None = None,
        name: str | None = None,
        speed: float | None = None,
    ) -> Any:
        # Generate speech using OpenAI
        raise NotImplementedError("generate_speech method not implemented")
