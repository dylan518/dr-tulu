from pathlib import Path
import sys

# Allow importing evaluation modules when running pytest from the `agent/` directory.
EVAL_DIR = (Path(__file__).parent.parent / "evaluation").resolve()
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from scholarqa_cs2_eval.scholarqa_cs2_eval import _extract_json_object  # noqa: E402


def test_extract_json_object_from_plain_json():
    obj = _extract_json_object('{"score": 7, "verdict": "PASS", "reasons": ["ok"]}')
    assert obj and obj["score"] == 7


def test_extract_json_object_from_fenced_block():
    text = """```json
{"score": 3, "verdict": "FAIL", "reasons": ["hallucinated"]}
```"""
    obj = _extract_json_object(text)
    assert obj and obj["verdict"] == "FAIL"


def test_extract_json_object_returns_none_for_garbage():
    assert _extract_json_object("no json here") is None


