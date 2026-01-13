from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from samplers import common
from samplers._types import Eval, SamplerBase, SingleEvalResult


JUDGE_SYSTEM = "You are a strict but fair grader for research QA."

JUDGE_PROMPT = """
You are grading a model answer to a ScholarQA-CSv2 research question.

Grade primarily on:
- Correctness: factual correctness and alignment with the question.
- Completeness: covers the key aspects expected by the question.
- Grounding: avoids hallucinations; uses citations when making factual claims.
- Clarity: coherent and readable.

If a reference answer is provided, prefer it for correctness.

Return ONLY valid JSON with this schema:
{{
  "score": <number from 0 to 10>,
  "verdict": "<PASS|FAIL>",
  "reasons": [<strings, short>]
}}

Question:
{question}

Reference answer (may be empty):
{reference_answer}

Model answer:
{model_answer}
""".strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a single JSON object from a judge response.
    """
    if not text:
        return None

    # Prefer a fenced json block if present.
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Otherwise extract the first {...} blob.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass
class ScholarQACS2Eval(Eval):
    """
    GPT-judge evaluation for ScholarQA-CSv2-style questions.

    This dataset is often distributed as a local JSONL file; the generation pipeline in this repo
    stores the original row under `row` or `original_data` depending on the caller.
    """

    grader_model: SamplerBase
    enable_checkpoint: bool = False
    n_threads: int = 10

    def _get_reference_answer(self, row: Dict[str, Any]) -> str:
        for k in ["reference_answer", "gold_answer", "answer", "target", "ground_truth"]:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def grade_one(self, row: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        question = (
            row.get("question")
            or row.get("problem")
            or row.get("query")
            or row.get("prompt")
            or ""
        )
        question = str(question).strip()

        ref = self._get_reference_answer(row)

        prompt = JUDGE_PROMPT.format(
            question=question,
            reference_answer=ref if ref else "(none)",
            model_answer=(response_text or "").strip(),
        )
        msg = [self.grader_model._pack_message(content=prompt, role="user")]
        judged = self.grader_model(msg).response_text

        obj = _extract_json_object(judged) or {}
        score = obj.get("score", None)
        verdict = obj.get("verdict", None)
        reasons = obj.get("reasons", None)

        # Coerce types safely
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        score_f = max(0.0, min(10.0, score_f))

        verdict_s = str(verdict or "").strip().upper()
        if verdict_s not in {"PASS", "FAIL"}:
            verdict_s = "PASS" if score_f >= 6.0 else "FAIL"

        if not isinstance(reasons, list):
            reasons = []
        reasons = [str(r)[:300] for r in reasons if str(r).strip()]

        return {
            "score": score_f,
            "verdict": verdict_s,
            "reasons": reasons,
            "raw_judge": judged,
        }

    def evaluate(self, generation_data: List[Dict[str, Any]], eval_save_dir: str = "eval_results") -> List[SingleEvalResult]:
        def _row_and_resp(ele: Dict[str, Any]) -> tuple[Dict[str, Any], str, str]:
            # Support both evaluation formats in this repo:
            # - evaluate.py converted format: {"row": ..., "response_text": ...}
            # - workflow output format: {"original_data": ..., "final_response": ...}
            row = ele.get("row") or ele.get("original_data") or {}
            response = ele.get("response_text") or ele.get("final_response") or ""
            ex_id = (
                ele.get("id")
                or ele.get("example_id")
                or (row.get("id") if isinstance(row, dict) else None)
                or "unknown"
            )
            return (row if isinstance(row, dict) else {}), str(response), str(ex_id)

        def _grade(ele: Dict[str, Any]) -> SingleEvalResult:
            row, response, ex_id = _row_and_resp(ele)
            graded = self.grade_one(row=row, response_text=response)
            return SingleEvalResult(
                id=ex_id,
                score=graded["score"],
                metrics={
                    "score": graded["score"],
                    "pass": 1.0 if graded["verdict"] == "PASS" else 0.0,
                },
                html=None,
                convo=None,
                example_level_metadata={
                    "verdict": graded["verdict"],
                    "reasons": graded["reasons"],
                    "raw_judge": graded.get("raw_judge", ""),
                    "full_traces": ele.get("full_traces"),
                },
                gt_answer=self._get_reference_answer(row) or None,
                pred_answer=response or None,
            )

        if self.enable_checkpoint:
            os.makedirs(eval_save_dir, exist_ok=True)
            checkpoint_path = os.path.join(eval_save_dir, "scholarqa_cs2_judge.checkpoint.json")
            return common.map_with_progress_checkpoint(
                _grade,
                generation_data,
                checkpoint_path,
                num_threads=self.n_threads,
                checkpoint_interval=max(10, self.n_threads),
            )

        return common.map_with_progress(_grade, generation_data, num_threads=self.n_threads)


