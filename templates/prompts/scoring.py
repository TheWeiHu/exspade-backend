def create_scoring_prompt(leaf_node: str, document: str) -> str:
    """
    Creates a prompt for scoring a document against a specific criterion.
    
    Parameters:
        leaf_node (str): The specific criterion to evaluate
        document (str): The document to evaluate
        
    Returns:
        str: A formatted prompt string that will elicit a numerical score from an LLM
    """
    return f"""{leaf_node}
---
{document}
---
Assign a score from 1 to 100 (with a mean of 75 and a standard deviation of 10).
Return just the score, with no additional annotations.""" 