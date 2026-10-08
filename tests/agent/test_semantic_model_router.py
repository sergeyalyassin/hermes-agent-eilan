from agent import semantic_model_router as router

def test_classify_coding():
    assert router.classify("Fix the Python bug in gateway/run_turn.py") == "coding"

def test_classify_reasoning():
    assert router.classify("Analyze the root cause and compare the architectural trade-offs.") == "reasoning"

def test_classify_light():
    assert router.classify("Hello") == "light"

def test_classify_general():
    assert router.classify("Explain the history of Sudanese migration to Egypt.") == "general"

def test_select_uses_free_routes(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    result = router.select("Fix this GitHub repository bug")
    assert result is not None
    assert result.name == "coding"
    assert result.model == "qwen/qwen3-coder:free"

def test_disabled_without_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_SEMANTIC_ROUTING", raising=False)
    assert router.select("Fix this Python bug") is None
