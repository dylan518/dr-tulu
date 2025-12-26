"""
LLM-based judge for ReSearch-style QA evaluation.

This module provides functions to judge whether a predicted answer is semantically
equivalent to a ground truth answer using an LLM, even when they are not textually identical.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Union

from .utils.run_utils import run_litellm

LOGGER = logging.getLogger(__name__)

# Template for LLM-based answer equivalence judgment
RE_SEARCH_LLM_JUDGE_TEMPLATE = """You are an expert at evaluating whether two answers are semantically equivalent.

Given a question and two answers, determine if they convey the same meaning or refer to the same entity/concept, even if phrased differently.

[Question]: {question}

[Predicted Answer]: {predicted_answer}

[Ground Truth Answer]: {ground_truth_answer}

Consider the following when judging:
- Minor spelling variations, abbreviations, or formatting differences should not affect equivalence
- Numerical answers should match exactly (or be mathematically equivalent)
- Named entities should refer to the same person, place, or thing
- Paraphrased answers that convey the same core meaning are equivalent

Your response must follow this exact format:
reasoning: <brief explanation of why the answers are equivalent or not>
equivalent: <yes or no>
""".strip()

# Precompile regex for answer extraction
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer_from_tags(prediction: str) -> Optional[str]:
    """
    Extract answer text from <answer>...</answer> tags.
    
    Args:
        prediction: The full model prediction string
        
    Returns:
        The extracted answer text, or None if no tags found or answer is empty
    """
    match = ANSWER_TAG_PATTERN.search(prediction)
    if not match:
        return None
    answer_text = match.group(1).strip()
    return answer_text if answer_text else None


def parse_labels(label: Union[str, List[str]]) -> List[str]:
    """
    Parse JSON label safely into a list of strings.
    
    Args:
        label: Either a JSON string or a list of labels
        
    Returns:
        List of label strings
    """
    try:
        parsed = json.loads(label) if isinstance(label, str) else label
        return parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, TypeError):
        if isinstance(label, list):
            return [str(l).strip() for l in label]
        else:
            return [str(label).strip()]


def judge_answer_equivalence(
    predicted_answer: str,
    ground_truth: str,
    question: str,
    grader_model: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Use LLM to judge if predicted answer matches ground truth.
    
    Args:
        predicted_answer: The extracted predicted answer
        ground_truth: The ground truth answer to compare against
        question: The original question (for context)
        grader_model: The LLM model to use for grading (default: env var or gpt-4.1-mini)
        
    Returns:
        Tuple of (score, reasoning) where score is 1.0 for equivalent, 0.0 otherwise
    """
    if grader_model is None:
        grader_model = os.environ.get("RE_SEARCH_JUDGE_MODEL", "gpt-4.1-mini")
    
    judge_prompt = RE_SEARCH_LLM_JUDGE_TEMPLATE.format(
        question=question,
        predicted_answer=predicted_answer,
        ground_truth_answer=ground_truth,
    )
    
    try:
        response = run_litellm(
            model_name=grader_model,
            system_prompt=None,
            user_prompt=judge_prompt,
        )
        
        # Parse the response for "equivalent: yes/no"
        match = re.search(r"equivalent:\s*(yes|no)", response, re.IGNORECASE)
        if match:
            is_equivalent = match.group(1).lower() == "yes"
            return (1.0 if is_equivalent else 0.0), response
        else:
            LOGGER.warning(f"Could not parse LLM judge response: {response}")
            return 0.0, response
            
    except Exception as e:
        LOGGER.error(f"LLM judge call failed: {e}")
        return 0.0, str(e)


def compute_re_search_llm_judge_reward(
    prediction: str,
    label: Union[str, List[str]],
    query: Optional[str] = None,
    grader_model: Optional[str] = None,
) -> Dict:
    """
    Compute the LLM-based judge reward for a ReSearch-style QA task.
    
    Args:
        prediction: The full model prediction (should contain <answer>...</answer> tags)
        label: Ground truth answer(s) - can be JSON string or list
        query: The original question (required for proper LLM judgment)
        grader_model: The LLM model to use for grading
        
    Returns:
        Dictionary containing:
        - 'score': The reward score (0.0 or 1.0)
        - 'extracted_answer': The extracted predicted answer
        - 'reasoning': The LLM's reasoning (if applicable)
        - 'error': Error message if any
    """
    result = {
        "score": 0.0,
        "extracted_answer": None,
        "reasoning": None,
        "error": None,
    }
    
    # 1. Parse labels
    parsed_labels = parse_labels(label)
    
    # 2. Extract answer between tags
    answer_text = extract_answer_from_tags(prediction)
    if not answer_text:
        result["error"] = "No <answer> tags found or answer is empty"
        return result
    
    result["extracted_answer"] = answer_text
    
    # 3. If no query provided, log warning
    if query is None:
        result["error"] = "No query provided for LLM judge - cannot perform semantic comparison"
        LOGGER.warning(result["error"])
        return result
    
    # 4. Try LLM judgment against each label, return best result
    best_score = 0.0
    best_reasoning = ""
    
    for lbl in parsed_labels:
        score, reasoning = judge_answer_equivalence(
            predicted_answer=answer_text,
            ground_truth=str(lbl),
            question=query,
            grader_model=grader_model,
        )
        if score > best_score:
            best_score = score
            best_reasoning = reasoning
        if score == 1.0:
            break  # Found a match, no need to check other labels
    
    result["score"] = best_score
    result["reasoning"] = best_reasoning
    
    return result


# For testing
if __name__ == "__main__":
    # Test the implementation
    test_prediction = """
    Let me think about this question.
    
    Based on my research, the answer is Paris.
    
    <answer>Paris</answer>
    """
    
    test_label = '["Paris", "paris, france"]'
    test_query = "What is the capital of France?"
    
    print("Testing compute_re_search_llm_judge_reward...")
    print(f"Prediction: {test_prediction}")
    print(f"Label: {test_label}")
    print(f"Query: {test_query}")
    print("-" * 50)
    
    result = compute_re_search_llm_judge_reward(
        prediction=test_prediction,
        label=test_label,
        query=test_query,
    )
    
    print(f"Result: {result}")

