import os
import asyncio
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from openai import AzureOpenAI

from .LLMAgentBase import LLMAgentBase, TokenUsage

# Load environment variables
load_dotenv()

class AzureOpenAIAgent(LLMAgentBase):
    def __init__(
        self,
        model: str = "gpt-4o",
        cache_dir: str = ".cache",
        cache_size_limit: int = 1024 * 1024 * 1024,  # 1GB
        requests_per_minute: int = 100,
    ):
        """
        Initialize Azure OpenAI Agent with caching and rate limiting.
        
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

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )

    async def _make_request(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, TokenUsage]:
        """Make request to Azure OpenAI"""
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
            print(f"Error in Azure OpenAI request: {str(e)}")
            raise 