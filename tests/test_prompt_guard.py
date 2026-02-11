import lifestyle_agent.core.prompt_guard as prompt_guard


def test_parse_guard_response_json_object():
    payload = prompt_guard._parse_guard_response(
        '{"violation": 1, "category": "System Exposure", "rationale": "override"}'
    )
    assert payload == {"violation": 1, "category": "System Exposure", "rationale": "override"}


def test_parse_guard_response_with_prefix():
    payload = prompt_guard._parse_guard_response(
        'Result: {"violation": 0, "category": null, "rationale": "ok"}'
    )
    assert payload["violation"] == 0


def test_build_decision_missing_violation_fail_closed():
    decision = prompt_guard._build_decision({"category": "x"}, error=None, fail_closed=True)
    assert decision.block is True
    assert decision.error == "missing_violation"


def test_coerce_violation_variants():
    assert prompt_guard._coerce_violation("1") is True
    assert prompt_guard._coerce_violation("safe") is False
