import os
import asyncio
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from .LLMAgentBase import LLMAgentBase, TokenUsage

# Load environment variables
load_dotenv()

class OpenAIAgent(LLMAgentBase):
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        cache_dir: str = ".cache",
        cache_size_limit: int = 1024 * 1024 * 1024,  # 1GB
        requests_per_minute: int = 100,
    ):
        """
        Initialize OpenAI Agent with caching and rate limiting.
        
        Args:
            model: The model to use for completions
            cache_dir: Directory to store response cache
            cache_size_limit: Maximum cache size in bytes
            requests_per_minute: Maximum requests allowed per minute
        """
        super().__init__(model, cache_dir, cache_size_limit, requests_per_minute)
        
        # OpenAI configuration
        self.api_key = os.getenv("OPENAI_API_KEY")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
        )

    async def _make_request(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, TokenUsage]:
        """Make request to OpenAI"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages
            )
            
            # Track token usage
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return response.choices[0].message.content, usage
        except Exception as e:
            print(f"Error in OpenAI request: {str(e)}")
            raise 