import os
from pathlib import Path
import sys

import pytest

# Allow importing evaluation modules when running pytest from the `agent/` directory.
EVAL_DIR = (Path(__file__).parent.parent / "evaluation").resolve()
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from samplers.sampler.chat_completion_sampler import ChatCompletionSampler  # noqa: E402
from scholarqa_cs2_eval.scholarqa_cs2_eval import (  # noqa: E402
    JUDGE_PROMPT,
    JUDGE_SYSTEM,
    _extract_json_object,
)


@pytest.mark.integration
def test_gpt41mini_judge_returns_json_if_key_configured():
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("AZURE_OPENAI_ENDPOINT"):
        pytest.skip("OPENAI_API_KEY (or Azure OpenAI env vars) not configured; skipping judge integration.")

    grader = ChatCompletionSampler(
        model=os.environ.get("SCHOLARQA_JUDGE_MODEL", "gpt-4.1-mini"),
        system_message=JUDGE_SYSTEM,
        temperature=0,
        max_tokens=300,
    )

    prompt = JUDGE_PROMPT.format(
        question="What is PageRank and why does it work?",
        reference_answer="",
        model_answer="PageRank is an algorithm used by search engines to rank web pages by importance. It models a random surfer and computes a stationary distribution over pages. It works because links act like votes, and the stationary distribution captures long-run visit probability.",
    )
    out = grader([grader._pack_message(role="user", content=prompt)]).response_text
    obj = _extract_json_object(out)
    assert obj is not None


