import asyncio
import json
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration


class _Trace(BaseModel):
    generated_text: str = ""


class _DummyWorkflow(BaseWorkflow):
    class Configuration(BaseWorkflowConfiguration):
        pass

    @property
    def _default_configuration_path(self):
        return None

    def setup_components(self) -> None:
        return

    async def __call__(self, problem: str, **kwargs):
        # Encode delay in the problem string: "text::0.25"
        delay_s = 0.05
        if "::" in problem:
            try:
                delay_s = float(problem.rsplit("::", 1)[-1])
            except Exception:
                delay_s = 0.05
        await asyncio.sleep(delay_s)
        return {
            "final_response": f"ok:{problem}",
            "full_traces": _Trace(generated_text=problem),
        }


@pytest.mark.asyncio
async def test_generate_dataset_batch_receiver_consumer_streams_to_jsonl(tmp_path: Path):
    # Build a tiny ScholarQA-CS2 JSONL locally (uses built-in loader path).
    ds_path = tmp_path / "scholarqa_cs2.jsonl"
    rows = [
        {"id": "a", "question": "qA::0.40"},
        {"id": "b", "question": "qB::0.10"},
        {"id": "c", "question": "qC::0.25"},
    ]
    ds_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    out_path = tmp_path / "out.jsonl"

    wf = _DummyWorkflow(skip_service_check=True)

    # Kick off generation but don't await completion immediately.
    task = asyncio.create_task(
        wf.generate_dataset_batch(
            dataset_config={"name": "scholarqa_cs2", "num_examples": 3, "local_path": str(ds_path)},
            batch_size=3,
            max_concurrent_tasks=2,
            output_file=str(out_path),
            receiver_consumer=True,
            include_original_data=False,
        )
    )

    # Within ~0.2s we should see at least one completed item (qB::0.10),
    # and the receiver/consumer should have appended it to the output file.
    t0 = time.time()
    wrote_early = False
    while time.time() - t0 < 1.0 and not task.done():
        if out_path.exists() and out_path.stat().st_size > 0:
            wrote_early = True
            break
        await asyncio.sleep(0.02)

    assert wrote_early, "expected receiver/consumer path to stream JSONL output before completion"

    results = await task
    assert len(results) == 3
    assert out_path.exists()
    # Final file should have 3 JSONL rows.
    lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3


