import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration


class _Trace(BaseModel):
    generated_text: str = ""
    model_input: dict | None = None


class _WF(BaseWorkflow):
    class Configuration(BaseWorkflowConfiguration):
        pass

    @property
    def _default_configuration_path(self):
        return None

    def setup_components(self) -> None:
        return

    async def __call__(self, problem: str, **kwargs):
        await asyncio.sleep(0.01)
        trace = _Trace(
            generated_text="ok",
            model_input={
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": "hi"},
                    {"role": "tool", "tool_call_id": "x", "name": "google_search", "content": "result"},
                    {"role": "assistant", "content": "final"},
                ]
            },
        )
        return {"final_response": "final", "full_traces": trace}


@pytest.mark.asyncio
async def test_generate_dataset_batch_messages_schema(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DR_OUTPUT_MESSAGES_SCHEMA", "1")

    ds_path = tmp_path / "scholarqa_cs2.jsonl"
    ds_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "question": "q1"}),
                json.dumps({"id": "b", "question": "q2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "out.jsonl"

    wf = _WF(skip_service_check=True)
    await wf.generate_dataset_batch(
        dataset_config={"name": "scholarqa_cs2", "num_examples": 2, "local_path": str(ds_path)},
        batch_size=2,
        max_concurrent_tasks=2,
        output_file=str(out_path),
        receiver_consumer=True,
        include_original_data=False,
    )

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 2
    for r in rows:
        assert set(r.keys()) >= {"example_id", "problem", "messages"}
        assert isinstance(r["messages"], list)
        assert r["messages"][0]["role"] == "system"


