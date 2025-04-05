def create_decomposition_prompt(
    user_document_format: str,
    user_question: str,
    user_requirements: list[str],
    additional_context: str,
) -> str:
    """
    Creates a prompt that instructs an LLM to decompose a complex question into weighted sub-questions.
    The prompt guides the LLM to either return a single 100% weighted prompt for simple questions,
    or break complex questions into multiple weighted prompts that sum to 100%.

    Parameters:
        user_document_format (str): The type of document being analyzed (e.g. "resumes")
        user_question (str): The main question to decompose
        user_requirements (list[str]): List of specific requirements or constraints
        additional_context (str): Any additional context needed for evaluation

    Returns:
        str: A formatted prompt string that will elicit weighted sub-questions from an LLM
    """
    requirements_bullets = "\n".join(f"- {req}" for req in user_requirements)
    additional_context = additional_context or "none"

    return f"""
You are a data engine responsible for analyzing a user's question and, if necessary, decomposing it into simpler sub-questions. Your objective is to produce a set of clear, standalone prompts that another agent will later use to evaluate the document. Follow these steps:

1. **Assess the User Provided Information**
    - **User Document Format:** {user_document_format}
    - **User Question:** {user_question}
    - **User Requirements:**
{requirements_bullets}
    - **Additional Context:** {additional_context}

2. **Analyze the Query**    
    - Determine if the query can be answered directly or if it needs to be broken down into multiple sub-questions.
        
3. **Determine the Approach**
    - If the query is simple and can be answered directly using the provided document, return a refined prompt in this format:
        - prompt; 100%
      Most questions that are trying to objectively determine if something is present or mentioned in the document should fall in this category.
    - If the query is complex, break it down into distinct aspects. For each aspect, assign an appropriate weight (as a percentage).
        
4. **For Complex Queries**
    - Identify each aspect that requires evaluation.
    - For each aspect, create a standalone prompt that explains how to assess that specific part. The prompts should be independent from each other.
    - Format each prompt on its own line as follows:
        - prompt; X%
        - prompt; Y%
    - Ensure every prompt includes all the necessary context so it can be understood independently. Do not enumerate the prompts.
    - Prompts should be phrased in a way where a higher score means better alignment with the user question.
        
5. **Additional Notes** 
    - The input document will be provided by another agent. Your prompts must be fully self-contained and should not reference any external instructions.
    - Every output line should match this regex: "-\s*(.*?);\s*(\d+)%"
""" 