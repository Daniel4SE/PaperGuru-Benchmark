"""Prompt templates used across §3.1 / §3.2 / §3.3 of the paper."""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Zero-shot (§3.1)
# ---------------------------------------------------------------------------
def zero_shot_prompt(question: str, answer_prefix: str = "Answer:") -> str:
    """Standard LM-Eval-Harness style zero-shot prompt.

    The harness uses "Question: {q}\\nAnswer:" with greedy/loglikelihood
    scoring, exactly the recipe the paper inherits (addendum).
    """
    return f"Question: {question}\n{answer_prefix}"


# ---------------------------------------------------------------------------
# Chain-of-Thought (§3.2)
# ---------------------------------------------------------------------------
# 8-shot CoT examples for GSM8K, replicated from Wei et al. 2022 / Wang et
# al. 2023 self-consistency, which §3.2 references as the prompt source.
DEFAULT_COT_FEW_SHOT_GSM8K: List[str] = [
    (
        "Q: There are 15 trees in the grove. Grove workers will plant trees "
        "in the grove today. After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?\n"
        "A: There are 15 trees originally. Then there were 21 trees after "
        "some more were planted. So there must have been 21 - 15 = 6. "
        "The answer is 6."
    ),
    (
        "Q: If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?\n"
        "A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. "
        "The answer is 5."
    ),
    (
        "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?\n"
        "A: Originally, Leah had 32 chocolates. Her sister had 42. "
        "So in total they had 32 + 42 = 74. After eating 35, they had "
        "74 - 35 = 39. The answer is 39."
    ),
    (
        "Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?\n"
        "A: Jason started with 20 lollipops. Then he had 12 after giving "
        "some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."
    ),
    (
        "Q: Shawn has five toys. For Christmas, he got two toys each from "
        "his mom and dad. How many toys does he have now?\n"
        "A: Shawn started with 5 toys. He got 2 from mom and 2 from dad, so "
        "4 in total. 5 + 4 = 9. The answer is 9."
    ),
    (
        "Q: There were nine computers in the server room. Five more "
        "computers were installed each day, from monday to thursday. How "
        "many computers are now in the server room?\n"
        "A: There were originally 9 computers. For each of 4 days, 5 more "
        "were added. So 5 * 4 = 20 were added. 9 + 20 = 29. The answer is 29."
    ),
    (
        "Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. "
        "On wednesday, he lost 2 more. How many golf balls did he have at "
        "the end of wednesday?\n"
        "A: Michael started with 58. After losing 23, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33. The answer is 33."
    ),
    (
        "Q: Olivia has $23. She bought five bagels for $3 each. How much "
        "money does she have left?\n"
        "A: Olivia had 23 dollars. 5 bagels for 3 dollars each is 5 * 3 = 15. "
        "23 - 15 = 8. The answer is 8."
    ),
]


DEFAULT_COT_FEW_SHOT_AQUA: List[str] = [
    (
        "Q: John found that the average of 15 numbers is 40. If 10 is added "
        "to each number then the mean of the numbers is?\n"
        "Options: A) 50, B) 45, C) 65, D) 78, E) 64\n"
        "A: If 10 is added to each number, the mean increases by 10. "
        "New mean = 40 + 10 = 50. The answer is A."
    ),
    (
        "Q: If a / b = 3/4 and 8a + 5b = 22, then find the value of a.\n"
        "Options: A) 1/2, B) 3/2, C) 5/2, D) 4/2, E) 7/2\n"
        "A: a/b = 3/4 means a = 3b/4. Substitute: 8(3b/4) + 5b = 22 "
        "→ 6b + 5b = 22 → 11b = 22 → b = 2. So a = 3*2/4 = 3/2. "
        "The answer is B."
    ),
    (
        "Q: A person is traveling at 20 km/hr and reached his destination "
        "in 2.5 hr. Find the distance.\n"
        "Options: A) 53, B) 55, C) 52, D) 60, E) 50\n"
        "A: Distance = speed * time = 20 * 2.5 = 50 km. The answer is E."
    ),
]


def cot_prompt(question: str, dataset: str = "gsm8k") -> str:
    """Build a few-shot CoT prompt followed by the new question."""
    if dataset == "gsm8k":
        examples = DEFAULT_COT_FEW_SHOT_GSM8K
    elif dataset == "aqua":
        examples = DEFAULT_COT_FEW_SHOT_AQUA
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    body = "\n\n".join(examples)
    return f"{body}\n\nQ: {question}\nA:"


# ---------------------------------------------------------------------------
# HumanEval (§3.3.1)
# ---------------------------------------------------------------------------
def humaneval_prompt(prompt: str) -> str:
    """HumanEval prompts are used verbatim -- they already include the
    function signature and a docstring (Chen et al. 2021)."""
    return prompt
