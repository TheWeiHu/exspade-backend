"""
SPADE is a tool for querying a set of documents with complex questions that decomposes them into a weighted tree of simpler questions.

after iterating on the execution plan using natural language, users can apply the plan to score all of the documents in parallel.

•⁠  ⁠given a job description and preferences, create an execution plan to rank these resumes
•⁠  ⁠given journal entries and a value system, rate how well each day went
•⁠  ⁠given a set of emails, and a set of OKR, decide which ones to prioritize
"""


import pandas as pd
import math
import asyncio
from pathlib import Path


from llm_agent import AzureOpenAIAgent
from tqdm import tqdm
from templates.prompts import create_scoring_prompt, create_feedback_prompt


class ExecutionPlan:
    def __init__(self, plan: str, documents: list[str] = []):
        self.plan = plan
        self.leaf_nodes = self.extract_leaf_nodes()
        self.documents = documents
        self.scores = None

    @classmethod
    def from_plan_string(cls, plan: str, documents: list[str] = []):
        return cls(cls._parse_plan(plan), documents)

    @classmethod
    def _parse_plan(cls, tree_string):
        """
        Given a string representation of a tree reconstructs the nested dictionary.

        Each line in the string should have the format:
            <effective_weight> - <prompt_text>
        with indentation (4 spaces per level) indicating the tree depth.

        Returns:
            dict: A nested dictionary where keys are tuples (prompt_text, node_weight)
                and values are subtrees (dictionaries). The node_weight is recovered using the parent's effective weight.
        """
        # The root dictionary that will be returned.
        root = {}
        # Stack holds tuples: (indent_level, parent_effective_weight, subtree_dictionary)
        # Start with a dummy root at level -1; its effective weight is 1.
        stack = [(-1, 1, root)]

        # Process each non-empty line.
        for line in tree_string.splitlines():
            if not line.strip():
                continue  # Skip empty lines
            # Determine the current indentation level (number of 4-space indents)
            indent_spaces = len(line) - len(line.lstrip(" "))
            level = indent_spaces // 4

            # Remove leading spaces and split the line into effective weight and prompt text.
            line_content = line.lstrip(" ")
            try:
                weight_str, prompt_text = line_content.split(" - ", 1)
            except ValueError:
                raise ValueError(f"Line format is incorrect: {line}")

            effective_weight = float(weight_str)

            # Find the parent by popping off levels deeper or equal to current level.
            while stack and stack[-1][0] >= level:
                stack.pop()
            if not stack:
                raise ValueError("Invalid tree string: inconsistent indentation.")

            # Parent's effective weight is from the top of the stack.
            parent_level, parent_effective_weight, parent_subtree = stack[-1]
            # Recover the original node weight.
            # (For the top-level nodes, parent's effective weight is 1.)
            node_weight = round(effective_weight / parent_effective_weight, 2)

            # Create the new node with an empty subtree.
            new_node = {}
            # Use the tuple (prompt_text, node_weight) as key.
            parent_subtree[(prompt_text, node_weight)] = new_node

            # Push the new node onto the stack.
            stack.append((level, effective_weight, new_node))

        return root

    def extract_leaf_nodes(self):
        """
        Extracts leaf nodes from the execution plan tree and maps them to their weights.

        Returns:
            dict: A dictionary mapping leaf node prompts to their weights
        """
        leaf_nodes = {}

        def traverse(tree, parent_weight=1.0):
            for (prompt, weight), subtree in tree.items():
                effective_weight = parent_weight * weight
                if not subtree:  # Leaf node
                    leaf_nodes[prompt] = effective_weight
                else:
                    traverse(subtree, effective_weight)

        traverse(self.plan)

        # Verify total weight of leaves sums to approximately 1
        total_weight = sum(leaf_nodes.values())
        if not math.isclose(total_weight, 1.0, rel_tol=0.01):
            raise ValueError(
                f"Total weight of leaves ({total_weight:.3f}) does not sum to 1"
            )

        return leaf_nodes

    async def execute_plan(self, model="gpt-4o"):
        # Create a task for each document.
        tasks = [
            asyncio.create_task(self._execute_plan_on_document(document, model))
            for document in self.documents
        ]
        # Run all tasks concurrently.
        results = await asyncio.gather(*tasks)
        # Return the results as a DataFrame.
        results = pd.DataFrame(results)
        self.scores = results
        return results

    async def _execute_plan_on_document(self, document: str, model: str):
        # TODO: need a way to sepecify the prior for the return distribution dynamically.
        async def process_leaf(leaf_node):
            prompt = create_scoring_prompt(leaf_node, document)
            async with AzureOpenAIAgent(model=model) as agent:
                return float(await agent.ask(prompt))

        # Create a list of tasks for all leaf nodes.
        tasks = [
            asyncio.create_task(process_leaf(leaf_node))
            for leaf_node in self.leaf_nodes.keys()
        ]

        # Gather all the task results concurrently.
        results_list = await asyncio.gather(*tasks)

        # Map the results back to their respective leaf nodes.
        results = dict(zip(self.leaf_nodes.keys(), results_list))
        return results
    

    async def modify_plan_through_user_feedback(self, feedback: str, model: str = "gpt-4"):
        """
        Updates the execution plan based on user feedback.

        Args:
            feedback (str): The user's feedback about how to modify the rubric

        Raises:
            Exception: If both attempts to generate a valid rubric fail
        """
        prompt = create_feedback_prompt(ExecutionPlan.plan_to_string(self.plan), feedback)
        async with AzureOpenAIAgent(model=model) as agent:
            result = await agent.ask(prompt)
            n_attempts = 3
            while n_attempts:
                try:
                    self.plan = self._parse_plan(result)
                    self.leaf_nodes = self.extract_leaf_nodes()
                    print(f"Attempt {n_attempts} succeeded")
                    break
                except Exception as e:
                    result = await agent.ask(" The error was: {e}. The format is not right or the weight does not add up to 1.0, re-attempt. Note: return only the final output, with no annotations")
                    n_attempts -= 1
                    print(result)
                    if n_attempts == 0:
                        raise Exception("Failed to parse plan after 5 attempts")

    @staticmethod
    def plan_to_string(tree, parent_weight=1, indent=""):
        """
        Recursively creates a string representation of a nested dictionary as a tree.
        Each node's effective weight is computed as:
        effective_weight = parent_weight * node_weight.

        The keys in the dictionary are tuples (prompt_text, weight) and leaf nodes have a key "prompt".

        Parameters:
            tree (dict): The nested dictionary representing the tree.
            parent_weight (float): The effective weight from the parent node.
            indent (str): The current indentation string.

        Returns:
            str: The string representation of the tree.
        """
        result = ""
        if isinstance(tree, dict):
            for key, subtree in tree.items():
                if isinstance(key, tuple):
                    prompt_text, weight = key
                    effective_weight = parent_weight * weight
                    result += indent + f"{effective_weight:.3f} - {prompt_text}\n"
                    # Recursively add the subtree's string using the effective weight as the new parent_weight
                    result += ExecutionPlan.plan_to_string(
                        subtree, effective_weight, indent + "    "
                    )
        return result

    def verify_weights(self, tolerance: float = 0.025) -> bool:
        """
        Verify that for each parent node, the sum of its children's weights equals the parent's weight within tolerance.
        For leaf nodes, verify that their total sum is within tolerance of 1.
        
        Args:
            tolerance (float): Maximum allowed difference from expected weight (default: 0.01)
            
        Returns:
            bool: True if all weight relationships are valid within tolerance, False otherwise
        """
        def verify_subtree(tree, parent_weight: float = 1.0) -> bool:
            if not tree:  # Leaf node
                return True
                
            children_weights = []
            for (_, weight), subtree in tree.items():
                effective_weight = parent_weight * weight
                children_weights.append(effective_weight)
                if not verify_subtree(subtree, effective_weight):
                    return False
                    
            if children_weights:  # If not a leaf node
                total_children_weight = sum(children_weights)
                if not math.isclose(total_children_weight, parent_weight, rel_tol=tolerance):
                    return False
            return True

        # First verify the tree structure
        if not verify_subtree(self.plan):
            return False
            
        # Then verify total leaf weights sum to 1
        total_leaf_weight = sum(self.leaf_nodes.values())
        return math.isclose(total_leaf_weight, 1.0, rel_tol=tolerance)


async def main():
    # Load test data
    plan = ExecutionPlan(
        Path("./test-data/litespace/plan.txt").read_text(),
        [f.read_text() for f in Path("./test-data/litespace/documents").iterdir() if f.is_file()]
    )

    # Test plan string serialization/deserialization
    plan_string = ExecutionPlan.plan_to_string(plan.plan)
    reparsed_plan = ExecutionPlan.from_plan_string(plan_string, plan.documents).plan
    print("Plan serialization works:", plan.plan == reparsed_plan)
    print("Initial weights valid:", plan.verify_weights())

    # Test plan execution
    results = await plan.execute_plan()
    results.to_csv("./test-data/litespace/tmp.csv", index=False)

    # Test weight modification
    plan.modify_plan_through_user_feedback(
        "Please increase the weight of technical skills and decrease the weight of soft skills"
    )
    print("Weights still valid after modification:", plan.verify_weights())
    print(ExecutionPlan.plan_to_string(plan.plan))
    plan.modify_plan_through_user_feedback(
        "Don't weigh overqualification as much."
    )
    print("Weights still valid after modification:", plan.verify_weights())
    print(ExecutionPlan.plan_to_string(plan.plan))


if __name__ == "__main__":
    asyncio.run(main())
