def create_feedback_prompt(current_plan: str, feedback: str) -> str:
    """
    Creates a prompt for updating a rubric based on user feedback.
    
    Parameters:
        current_plan (str): The current rubric structure
        feedback (str): User feedback about how to modify the rubric
        
    Returns:
        str: A formatted prompt string that will elicit an updated rubric from an LLM
    """
    return f"""
You are given a rubric structured as a tree along with user feedback. Update the rubric based on the following instructions:
1. Feel free to modify the rubric prompts to better align with the user feedback.
2. Try to coalesce duplicate prompts or prompts that are very similar.
3. Do not reduce the depth of the tree, but do remove prompts that have a weight of 0.
4. It is important when adjusting the weights that, in the end, the total weight of the leaves add up to 1.0.
5. Try to not change the vocabulary and phrasing too aggressively, if not necessary.

Return only the updated rubric in the same tree format without any additional annotations, formating, or explanations.

Rubric:
{current_plan}

User Feedback:
{feedback}""" 