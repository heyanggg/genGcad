from __future__ import annotations

from collections.abc import Callable


class OriginalSmartGenAPIBackend:
    """Compatibility adapter around SmartGen's retained ``LLM_call`` path.

    It is intentionally never selected by the source-only experiment. A caller
    must explicitly supply the official call function and its configured client.
    """

    backend_type = "original_api"
    requires_api_key = True

    def __init__(self, llm_call: Callable[[object, str], str], client: object):
        self._llm_call = llm_call
        self._client = client

    def generate(self, prompt: str) -> str:
        return self._llm_call(self._client, prompt)
