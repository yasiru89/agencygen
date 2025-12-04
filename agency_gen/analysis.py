"""
Task analysis helpers to choose the best agent pattern.
"""

import json
from typing import Dict, Any, List

from google.adk.agents import LlmAgent

from .config import DEFAULT_MODEL


def _analyze_task_keywords(task: str, allow_composite: bool = True) -> Dict[str, Any]:
    """
    Analyze a task using keyword matching (fast, no API call).
    Returns pattern and reasoning based on keyword detection.
    For composite patterns, also returns sub_patterns list.
    """
    task_lower = task.lower()

    detected_patterns: List[Any] = []

    # RLM detection - check first as it may override other patterns
    rlm_chunking_keywords = [
        "long document",
        "large text",
        "entire book",
        "full transcript",
        "complete file",
        "process all",
        "summarize the entire",
    ]
    rlm_iterative_keywords = [
        "refine until",
        "iterate",
        "improve repeatedly",
        "perfect",
        "polish until",
        "recursive",
        "self-improve",
    ]
    rlm_hierarchical_keywords = [
        "break down",
        "decompose",
        "step by step",
        "sub-problems",
        "divide and conquer",
        "hierarchical",
        "recursive analysis",
    ]

    # Detect RLM patterns
    if any(kw in task_lower for kw in rlm_chunking_keywords) or len(task) > 8000:
        detected_patterns.append(("rlm_chunking", "long context processing"))
    elif any(kw in task_lower for kw in rlm_iterative_keywords):
        detected_patterns.append(("rlm_iterative", "iterative self-refinement"))
    elif any(kw in task_lower for kw in rlm_hierarchical_keywords):
        detected_patterns.append(("rlm_hierarchical", "hierarchical decomposition"))

    reliability_keywords = [
        "calculate",
        "math",
        "compute",
        "exact",
        "precise",
        "accurate",
        "correct",
        "factual",
        "verify",
    ]
    if any(kw in task_lower for kw in reliability_keywords):
        detected_patterns.append(("majority_voting", "accuracy/calculation"))

    quality_keywords = [
        "write",
        "compose",
        "draft",
        "create",
        "essay",
        "email",
        "letter",
        "story",
        "article",
        "polish",
        "professional",
    ]
    if any(kw in task_lower for kw in quality_keywords):
        detected_patterns.append(("reflection", "quality writing"))

    debate_keywords = [
        "analyze",
        "compare",
        "pros and cons",
        "debate",
        "argue",
        "perspective",
        "opinion",
        "ethics",
        "should",
    ]
    if any(kw in task_lower for kw in debate_keywords):
        detected_patterns.append(("debate", "multiple perspectives"))

    composite_keywords = [
        "then",
        "and then",
        "after that",
        "first",
        "second",
        "finally",
        "steps",
        "multi-step",
        "pipeline",
    ]
    has_composite_indicators = any(kw in task_lower for kw in composite_keywords)

    if allow_composite and (
        len(detected_patterns) >= 2
        or (has_composite_indicators and len(detected_patterns) >= 1)
    ):
        if len(detected_patterns) == 1:
            detected_patterns.append(("single_agent", "general task handling"))

        sub_patterns = [p[0] for p in detected_patterns]
        reasons = [p[1] for p in detected_patterns]
        return {
            "pattern": "composite",
            "reasoning": f'Complex task requiring: {", ".join(reasons)} - using composite pattern',
            "sub_patterns": sub_patterns,
        }

    if len(detected_patterns) == 1:
        return {
            "pattern": detected_patterns[0][0],
            "reasoning": f"Task requires {detected_patterns[0][1]} - using {detected_patterns[0][0]}",
        }

    if not allow_composite and len(detected_patterns) > 1:
        primary = detected_patterns[0]
        return {
            "pattern": primary[0],
            "reasoning": f"Composite disabled; prioritizing {primary[1]} with {primary[0]}",
        }

    return {
        "pattern": "single_agent",
        "reasoning": "Straightforward task - using single focused agent",
    }


async def _analyze_task_llm(
    task: str,
    model: str = DEFAULT_MODEL,
    allow_composite: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a task using an LLM (smarter, uses API call).
    """
    from .runner import run_agent  # Local import to avoid circular dependency

    analyzer_instruction = """You analyze tasks and choose the best agent pattern.

Available patterns:
- single_agent: For simple, straightforward tasks
- majority_voting: For tasks requiring accuracy/reliability (math, facts)
- reflection: For tasks requiring quality/polish (writing, creative)
- debate: For tasks benefiting from multiple perspectives (analysis, ethics)
- composite: For complex multi-step tasks requiring MULTIPLE patterns combined
- rlm_chunking: For very long inputs that need to be processed in chunks
- rlm_iterative: For tasks requiring iterative self-refinement until convergence
- rlm_hierarchical: For complex problems that should be recursively decomposed

For composite, also specify which sub_patterns to combine (2-3 patterns from above).
For RLM patterns, use them when the task explicitly needs recursion or long-context handling."""
    if not allow_composite:
        analyzer_instruction += """
IMPORTANT: Composite patterns are DISABLED for this run. Choose the single best pattern.
Do not return "composite". If multiple patterns seem relevant, pick the primary one."""
    analyzer_instruction += """
Respond with ONLY a JSON object (no markdown):
{"pattern": "pattern_name", "reasoning": "one sentence why"}

For composite patterns:
{"pattern": "composite", "reasoning": "why", "sub_patterns": ["pattern1", "pattern2"]}

Examples:
Task: "What is 15% of 240?"
{"pattern": "majority_voting", "reasoning": "Math calculation benefits from voting for accuracy"}

Task: "Write a cover letter"
{"pattern": "reflection", "reasoning": "Writing task benefits from self-critique for quality"}

Task: "Should I buy or rent a house?"
{"pattern": "debate", "reasoning": "Decision benefits from exploring multiple perspectives"}

Task: "What is the capital of France?"
{"pattern": "single_agent", "reasoning": "Simple factual question needs one focused answer"}

Task: "Calculate the ROI and then write a professional report analyzing the investment"
{"pattern": "composite", "reasoning": "Requires accurate calculation AND quality writing", "sub_patterns": ["majority_voting", "reflection"]}

Task: "First analyze the pros and cons of remote work, then write a persuasive essay"
{"pattern": "composite", "reasoning": "Needs debate for analysis AND reflection for quality writing", "sub_patterns": ["debate", "reflection"]}"""

    analyzer = LlmAgent(
        name="task_analyzer",
        model=model,
        instruction=analyzer_instruction,
    )

    response = await run_agent(analyzer, f"Task: {task}")

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        result = json.loads(cleaned)

        valid_patterns = [
            "single_agent",
            "majority_voting",
            "reflection",
            "debate",
            "composite",
            "rlm_chunking",
            "rlm_iterative",
            "rlm_hierarchical",
        ]
        if result.get("pattern") not in valid_patterns:
            result["pattern"] = "single_agent"
            result["reasoning"] = "LLM returned invalid pattern - defaulting to single agent"

        if result.get("pattern") == "composite":
            sub_patterns = result.get("sub_patterns", [])
            valid_sub = [
                "single_agent",
                "majority_voting",
                "reflection",
                "debate",
            ]
            sub_patterns = [p for p in sub_patterns if p in valid_sub]
            if len(sub_patterns) < 2:
                result["pattern"] = "single_agent"
                result["reasoning"] = "Composite needs 2+ sub-patterns - defaulting to single agent"
            else:
                result["sub_patterns"] = sub_patterns

        if not allow_composite and result.get("pattern") == "composite":
            fallback = (result.get("sub_patterns") or ["single_agent"])[0]
            result = {
                "pattern": fallback,
                "reasoning": f"Composite disabled; using {fallback} instead of composite",
            }

        return result
    except (json.JSONDecodeError, KeyError):
        return _analyze_task_keywords(task, allow_composite=allow_composite)

