"""
The pipeline from user entry to execution plan is as follows:
- 
"""

import asyncio
import re

from ExecutionPlan import ExecutionPlan
from LLMAgent import LLMAgent
from templates.prompts import create_decomposition_prompt


class ExecutionPlanGenerator:
    def __init__(
        self,
        user_document_format,
        user_question,
        user_requirements,
        additional_context,
        max_depth=3,
    ):
        """
        Initializes the ExecutionPlanGenerator with the necessary parameters.

        Parameters:
            user_document_format (str): Format of the user's document.
            user_question (str): The question provided by the user.
            user_requirements (list): List of requirement strings.
            additional_context (str): Additional context provided.
            max_depth (int): Maximum recursion depth.
        """
        self.user_document_format = user_document_format
        self.user_question = user_question
        self.user_requirements = user_requirements
        self.additional_context = additional_context
        self.max_depth = max_depth

    async def process(self, model="o3-mini", depth=0):
        """
        Processes the prompt by calling generate_weighted_prompts with the current parameters,
        extracts the keys from the output, and recursively processes each key as a new question.

        Parameters:
            depth (int): Current recursion depth.

        Returns:
            dict: A nested dictionary representing the prompt and any recursive sub-prompts.
        """
        if depth >= self.max_depth:
            return {}

        # Create the prompt for the current input
        weight_dict = generate_weighted_prompts(
            self.user_document_format,
            self.user_question,
            self.user_requirements,
            self.additional_context,
            model
        )

        result = {}
        tasks = {}

        # For each extracted key, process recursively in parallel if needed.
        for key, weight in weight_dict.items():
            if weight == 1:
                result[(key, weight)] = {}
            else:
                # Create a new instance with the new question (the extracted key)
                new_generator = ExecutionPlanGenerator(
                    self.user_document_format,
                    key,  # new user_question is the extracted key
                    [],  # propagate an empty requirements list (or adjust as needed)
                    self.additional_context,
                    self.max_depth,
                )
                tasks[(key, weight)] = asyncio.create_task(
                    new_generator.process(model, depth=depth + 1)
                )

        # Await all asynchronous tasks concurrently
        for k, task in tasks.items():
            result[k] = await task

        return result


def generate_weighted_prompts(
    user_document_format, user_question, user_requirements, additional_context, model="o3-mini"
):
    """
    Generates a dictionary mapping evaluation prompts to their weights by querying an LLM.

    The function creates a prompt asking the LLM to break down a complex question into
    weighted sub-prompts. It validates that the weights sum to 1 and retries on failure.

    Args:
        user_document_format (str): The format of documents being evaluated (e.g. "resumes")
        user_question (str): The main evaluation question
        user_requirements (list): List of specific requirements or constraints
        additional_context (str): Any additional context needed for evaluation

    Returns:
        dict: Mapping of evaluation prompts to their decimal weights that sum to 1.0

    Raises:
        Exception: If unable to generate valid weighted prompts after 5 attempts
    """
    prompt = create_decomposition_prompt(
        user_document_format, user_question, user_requirements, additional_context
    )

    agent = LLMAgent(model=model)
    weights = agent.ask(prompt)

    n_attempts = 3
    while n_attempts:
        try:
            weight_dict = parse_weighted_prompts(weights)
            if sum(weight_dict.values()) == 1:
                return weight_dict
            else:
                print(weights)
                weights = agent.ask(
                    "the weights do not sum to 1. note: return only the final output, with no annotations."
                )
        except Exception:
            weights = agent.ask(
                "the format is not right, re-attempt. note: return only the final output, with no annotations."
            )
            print(weights)
            n_attempts -= 1
    raise Exception


def parse_weighted_prompts(text: str) -> dict:
    """
    Extracts key-value pairs from the provided text where each non-empty line must be in the format:
    "- <query>, <percentage>%". The function returns a dictionary with the query as the key
    and the percentage as a float (e.g., 95% becomes 0.95). If any non-empty line does not match
    the expected format after trimming, a ValueError is raised.

    Args:
        text (str): Multiline string containing the queries and percentages.

    Returns:
        dict: A dictionary mapping each query to its percentage as a float.

    Raises:
        ValueError: If a non-empty line does not match the required format.
    """
    # Regular expression pattern to match the required format for each line
    pattern = r"\s*(.*?);\s*(\d+)%"
    result = {}

    # Process each line individually
    for line in text.splitlines():
        trimmed_line = line.strip()
        if not trimmed_line:
            continue  # Skip empty lines

        # Ensure the entire trimmed line matches the pattern
        match = re.fullmatch(pattern, trimmed_line)
        if not match:
            raise ValueError(f"Line does not match required format: {trimmed_line}")

        query, percent = match.groups()
        result[query.strip()] = float(percent) / 100

    return result


async def main():
    import json

    user_input = json.load(open("./test-data/litespace/user_input.json"))
    gen = ExecutionPlanGenerator(
        user_input["user_doc_format"],
        user_input["user_question"],
        user_input["user_requirements"],
        user_input["additional_context"],
        max_depth=1,
    )
    result = await gen.process()
    p = ExecutionPlan(result)
    print(ExecutionPlan.plan_to_string(p.plan))


if __name__ == "__main__":
    asyncio.run(main())
