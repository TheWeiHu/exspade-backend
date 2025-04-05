def create_scoring_prompt(leaf_node: str, document: str) -> str:
    return f"""{leaf_node}
---
{document}
---
Assign a score from 1 to 100 (with a mean of 75 and a standard deviation of 10).
Return just the score, with no additional annotations.""" 