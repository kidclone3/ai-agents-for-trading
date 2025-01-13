from typing import Any, List, Optional

import groq

from src.ai.base_provider import AIMessage, AIResponse, BaseAIProvider


DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider(BaseAIProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.async_client: groq.AsyncGroq = groq.AsyncClient(api_key)

    async def generate_text(
        self,
        messages: List[AIMessage],
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AIResponse:
        formatted_messages = [msg.to_dict() for msg in messages]
        chat_completions = await self.async_client.chat.completions.create(
            formatted_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return AIResponse(
            completions=chat_completions.completions,
            model=chat_completions.model,
            temperature=chat_completions.temperature,
            max_tokens=chat_completions.max_tokens,
        )

    async def generate_speech(
        self,
        text: str,
        model: str | None = None,
        name: str | None = None,
        speed: float | None = None,
    ) -> Any:
        # Generate speech using OpenAI
        raise NotImplementedError("generate_speech method not implemented")
