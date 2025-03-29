import os
import asyncio
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from functools import partial

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")
env = os.getenv("ENV")

client = OpenAI(api_key=api_key)


def query_llm(prompt, model="o3-mini", messages=[]):
    if env == "dev":  # don't bother calling API in dev mode.
        return ""
    stream = client.chat.completions.create(
        model=model,
        messages=messages
        + [
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
    return result


def query_llm_high(prompt, messages=[]):
    if env == "dev":  # don't bother calling API in dev mode.
        return ""
    stream = client.chat.completions.create(
        model="o3-mini",
        messages=messages
        + [
            {"role": "user", "content": prompt},
        ],
        stream=True,
        reasoning_effort="high",
    )
    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
    return result


def query_llm(prompt, model="o3-mini", messages=[]):
    if env == "dev":
        return ""
    stream = client.chat.completions.create(
        model=model,
        messages=messages + [{"role": "user", "content": prompt}],
        stream=True,
    )
    result = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
    return result


async def query_llm_async(prompt, model="o3-mini", messages=[]):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(query_llm, prompt, model, messages))


async def query_llm_high_async(prompt, messages=[]):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(query_llm_high, prompt, messages))


def create_folder_if_not_exists(directory):
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def save_text_file(file_path, content):
    """
    Saves the provided text content to the specified file path.

    Parameters:
    file_path (str): The complete file path where the text file should be saved.
    content (str): The text content to save in the file.
    """
    # Extract the directory from the file path
    directory = os.path.dirname(file_path)
    create_folder_if_not_exists(directory)

    # Write the content to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)
