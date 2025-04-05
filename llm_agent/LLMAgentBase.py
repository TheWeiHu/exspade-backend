from abc import ABC, abstractmethod
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import diskcache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TokenUsage:
    """Track token usage and cache hits"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached: bool = False


class LLMAgentBase(ABC):
    """Base class for LLM agents with shared functionality"""
    
    def __init__(
        self,
        model: str,
        cache_dir: str = ".cache",
        cache_size_limit: int = 1024 * 1024 * 1024,  # 1GB
        requests_per_minute: int = 100,
    ):
        self.model = model
        self.requests_per_minute = requests_per_minute
        self.token_usage = []
        self.semaphore = asyncio.Semaphore(requests_per_minute)
        self.env = os.getenv("ENV")
        
        # Set up disk cache
        self.cache = diskcache.Cache(cache_dir, size_limit=cache_size_limit)

        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []

    def _generate_cache_key(self, prompt: str, messages: List[Dict[str, str]]) -> str:
        """Generate a unique cache key for the request"""
        key_data = {
            "prompt": prompt,
            "messages": messages,
            "model": self.model
        }
        return json.dumps(key_data, sort_keys=True)

    @abstractmethod
    async def _make_request(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, TokenUsage]:
        """Make the actual request to the LLM provider"""
        pass

    async def _rate_limited_request(
        self, 
        prompt: str, 
        messages: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, TokenUsage]:
        """Execute rate-limited request to LLM provider"""
        async with self.semaphore:
            # Add delay if we're approaching rate limit
            await asyncio.sleep(60 / self.requests_per_minute)
            
            try:
                # Create the new user message
                user_message = {"role": "user", "content": prompt}
                
                # Combine conversation history with any additional messages
                current_messages = self.conversation_history.copy()
                if messages is not None:
                    current_messages.extend(messages)
                current_messages.append(user_message)
                
                return await self._make_request(current_messages)
            except Exception as e:
                print(f"Error in LLM request: {str(e)}")
                raise

    async def ask(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        bypass_cache: bool = False,
        clear_history: bool = False
    ) -> str:
        """
        Send a request to LLM with caching and rate limiting.
        
        Args:
            prompt: The prompt to send
            messages: Optional list of previous messages for context
            bypass_cache: If True, skip cache lookup
            clear_history: If True, clear conversation history before this request
            
        Returns:
            The model's response text
        """
        if self.env == "dev":
            return ""

        if clear_history:
            self.conversation_history = []

        # Create the new user message
        user_message = {"role": "user", "content": prompt}
        
        # Combine conversation history with any additional messages
        current_messages = self.conversation_history.copy()
        if messages is not None:
            current_messages.extend(messages)
        current_messages.append(user_message)

        cache_key = self._generate_cache_key(prompt, current_messages)

        # Check cache first
        if not bypass_cache:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                response_text, usage = cached_response
                usage.cached = True
                self.token_usage.append(usage)
                # Add to conversation history even if cached
                self.conversation_history.append(user_message)
                self.conversation_history.append({"role": "assistant", "content": response_text})
                return response_text

        # Get response from API with rate limiting
        response_text, usage = await self._rate_limited_request(prompt, messages)
        
        # Update conversation history
        self.conversation_history.append(user_message)
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Cache the response
        self.cache.set(cache_key, (response_text, usage))
        self.token_usage.append(usage)
        
        return response_text

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history"""
        return self.conversation_history.copy()

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about token usage and cache hits"""
        total_requests = len(self.token_usage)
        cache_hits = sum(1 for usage in self.token_usage if usage.cached)
        total_tokens = sum(usage.total_tokens for usage in self.token_usage)
        
        return {
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total_requests if total_requests > 0 else 0,
            "total_tokens": total_tokens,
            "average_tokens_per_request": total_tokens / total_requests if total_requests > 0 else 0
        }

    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()

    def get_cache_size(self) -> int:
        """Get current cache size in bytes"""
        return self.cache.size

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cache.close() 