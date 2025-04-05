import os
import asyncio
from typing import List, Dict, Tuple
import diskcache
from dotenv import load_dotenv
from openai import AzureOpenAI

from .base import BaseLLMAgent, TokenUsage

# Load environment variables
load_dotenv()

class AzureLLMAgent(BaseLLMAgent):
    def __init__(
        self,
        model: str = "gpt-4",
        cache_dir: str = ".cache",
        cache_size_limit: int = 1024 * 1024 * 1024,  # 1GB
        requests_per_minute: int = 100,
    ):
        """
        Initialize Azure LLM Agent with caching and rate limiting.
        
        Args:
            model: The model to use for completions
            cache_dir: Directory to store response cache
            cache_size_limit: Maximum cache size in bytes
            requests_per_minute: Maximum requests allowed per minute
        """
        super().__init__(model, cache_dir, cache_size_limit, requests_per_minute)
        
        # Azure OpenAI configuration
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.env = os.getenv("ENV")

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )

        # Set up disk cache
        self.cache = diskcache.Cache(cache_dir, size_limit=cache_size_limit)

    async def _make_request(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, TokenUsage]:
        """Make request to Azure OpenAI"""
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

    async def ask(
        self,
        prompt: str,
        messages: List[Dict[str, str]] = [],
        bypass_cache: bool = False
    ) -> str:
        """
        Send a request to Azure OpenAI with caching and rate limiting.
        
        Args:
            prompt: The prompt to send
            messages: List of previous messages for context
            bypass_cache: If True, skip cache lookup
            
        Returns:
            The model's response text
        """
        if self.env == "dev":
            return ""

        cache_key = self._generate_cache_key(prompt, messages)

        # Check cache first
        if not bypass_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                response_text, usage = cached_response
                usage.cached = True
                self.token_usage.append(usage)
                return response_text

        # Get response from API with rate limiting
        response_text, usage = await self._make_request(messages)
        
        # Cache the response
        self.cache.set(cache_key, (response_text, usage))
        self.token_usage.append(usage)
        
        return response_text

    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()

    def get_cache_size(self) -> int:
        """Get current cache size in bytes"""
        return self.cache.size

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cache.close() 