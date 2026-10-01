import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import error_handling as eh


def make_state(max_retries=3):
  return {
    "messages": [HumanMessage(content="Hello!")],
    "error": None,
    "retry_count": 0,
    "max_retries": max_retries,
    "success": False,
  }


def test_success_first_try():
  with patch.object(eh.random, "random", return_value=0.9), patch.object(eh, "LLM") as llm:
    llm.invoke.return_value = AIMessage(content="hi")
    r = eh.create_robust_agent().invoke(make_state())
  assert r["success"] and r["retry_count"] == 0
  assert r["messages"][-1].content == "hi"


def test_retry_then_success():
  # first two calls hit the injected failure, third succeeds
  with patch.object(eh.random, "random", side_effect=[0.1, 0.1, 0.9]), patch.object(eh, "LLM") as llm:
    llm.invoke.return_value = AIMessage(content="hi")
    r = eh.create_robust_agent().invoke(make_state())
  assert r["success"] and r["retry_count"] == 2


def test_llm_always_fails_goes_to_handle_error():
  with patch.object(eh.random, "random", return_value=0.9), patch.object(eh, "LLM") as llm:
    llm.invoke.side_effect = RuntimeError("API down")
    r = eh.create_robust_agent().invoke(make_state(max_retries=3))
  assert not r["success"]
  assert r["retry_count"] == 3
  assert llm.invoke.call_count == 3
  assert "API down" in r["messages"][-1].content


@pytest.mark.parametrize("max_retries", [1, 2, 5])
def test_respects_max_retries(max_retries):
  with patch.object(eh.random, "random", return_value=0.9), patch.object(eh, "LLM") as llm:
    llm.invoke.side_effect = RuntimeError("boom")
    r = eh.create_robust_agent().invoke(make_state(max_retries=max_retries))
  assert r["retry_count"] == max_retries
