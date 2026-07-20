from SmartGen.generation_backends.original_api import OriginalSmartGenAPIBackend


def test_original_api_adapter_requires_explicit_client_and_call():
    calls = []

    def call(client, prompt):
        calls.append((client, prompt))
        return "response"

    client = object()
    backend = OriginalSmartGenAPIBackend(call, client)
    assert backend.requires_api_key is True
    assert backend.generate("prompt") == "response"
    assert calls == [(client, "prompt")]
