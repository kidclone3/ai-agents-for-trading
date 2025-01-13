from typing import Any, List, Optional
from src.agents.ai.base_provider import (
    AIMessage,
    AIResponse,
    BaseAIProvider,
)
import openai

DEFAULT_VOICE_MODEL = "tts-1"  # or tts-1-hd for higher quality
DEFAULT_VOICE_NAME = "nova"  # Options: alloy, echo, fable, onyx, nova, shimmer
DEFAULT_VOICE_SPEED = 1  # 0.25 to 4.0


class OpenAIProvider(BaseAIProvider):
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
        pass

    async def generate_speech(
        self,
        text: str,
        model: str = DEFAULT_VOICE_MODEL,
        name: str = DEFAULT_VOICE_NAME,
        speed: float = DEFAULT_VOICE_SPEED,
    ) -> Any:
        # Generate speech using OpenAI

        response = openai.audio.speech.create(
            model=model, voice=name, speed=speed, input=text
        )
        return response
