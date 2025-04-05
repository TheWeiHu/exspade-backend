import logging
import asyncio
from functools import partial
import os
from logging.handlers import RotatingFileHandler

from utils import query_llm, query_llm_high
from Logger import setup_logger


# TODO: there needs to be a way to specify the expected return type, and to reiterat if it doesn't match.


class LLMAgent:
    """
    An agent class that manages a conversation history between the user and the system,
    and uses the query_llm function to generate responses. The ask method now allows
    specifying how many recent messages to include in the conversation history when querying.
    """

    def __init__(self, model="o3-mini", env="prod"):
        """Initializes the LLM agent with an empty conversation history."""
        self.history = []  # List to keep the conversation history as dicts
        self.model = model
        self.env = env
        
        # Set up logging
        self.logger = setup_logger(
            log_group_name=f'/spade/backend/llm_{env}',
            log_stream_name=f'llm_prompts_{env}'
        )

    def add_message(self, role: str, content: str):
        """Adds a message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def log_prompt(self, prompt: str):
        """Logs the user prompt using the configured logger."""
        try:
            self.logger.info(f"User Prompt: {prompt}")
        except Exception as e:
            print(f"Logging error: {e}")

    def ask(
        self, prompt: str, history_limit: int = 0, high_effort: bool = False
    ) -> str:
        """
        Sends a user prompt to the LLM and returns the system's response.
        The conversation history (both user and system messages) is maintained.
        The 'history_limit' parameter specifies how many of the most recent messages
        to include in the query. If None, the entire history is used.
        Additionally, logs the prompt to a local 'log.txt' file.

        Args:
            prompt (str): The user's prompt.
            history_limit (int, optional): Number of recent messages to include. Defaults to 0.

        Returns:
            str: The LLM's response.
        """
        # Log the prompt to file
        self.log_prompt(prompt)

        # Add the user prompt to history
        self.add_message("user", prompt)

        # Determine which portion of the conversation history to send
        messages_to_send = (
            self.history[-history_limit:]
            if history_limit and history_limit > 0
            else self.history
        )

        # Query the LLM with retries
        max_retries = 5
        retry_delay = 1  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                # Query the LLM with the selected conversation history
                if high_effort:
                    response = query_llm_high(prompt, messages=messages_to_send)
                else:
                    response = query_llm(
                        prompt, model=self.model, messages=messages_to_send
                    )

                # Add the system response to history
                self.add_message("assistant", response)
                return response

            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise  # Re-raise the last exception

                # Exponential backoff
                wait_time = retry_delay * (2**attempt)
                asyncio.sleep(wait_time)

    async def ask_async(
        self, prompt: str, history_limit: int = 0, high_effort: bool = False
    ):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self.ask, prompt, history_limit, high_effort)
        )

    def get_history(self) -> list:
        """Returns the full conversation history."""
        return self.history

    def clear_history(self):
        """Clears the conversation history."""
        self.history = []

