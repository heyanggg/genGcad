from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


class MissingCodexResponse(FileNotFoundError):
    pass


class CodexFileBackend:
    """Offline bridge between SmartGen prompts and Codex-authored response files."""

    def __init__(self, directory: str | Path, mode: str = "consume"):
        if mode not in {"export", "consume"}:
            raise ValueError("mode must be export or consume")
        self.directory = Path(directory)
        self.mode = mode
        self.prompts = self.directory / "prompts"
        self.responses = self.directory / "responses"
        self.prompts.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_id(request_id: object) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(request_id)).strip("._")
        if not value:
            raise ValueError("request_id contains no safe filename characters")
        return value

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _update_manifest(self, request_id: str, prompt: str) -> None:
        path = self.directory / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
            "backend": "codex_file",
            "external_api_used": False,
            "requests": {},
        }
        payload["requests"][request_id] = {
            "prompt": f"prompts/{request_id}.txt",
            "response": f"responses/{request_id}.txt",
            "prompt_sha256": self._sha256(prompt),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def generate(self, request_id: object, prompt: str) -> str | None:
        safe_id = self._safe_id(request_id)
        prompt_path = self.prompts / f"{safe_id}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        self._update_manifest(safe_id, prompt)
        if self.mode == "export":
            return None
        response_path = self.responses / f"{safe_id}.txt"
        if not response_path.is_file():
            raise MissingCodexResponse(
                f"Codex response missing: {response_path}. Run --codex-mode export, author the response, then consume."
            )
        response = response_path.read_text(encoding="utf-8").strip()
        if not response:
            raise ValueError(f"Codex response is empty: {response_path}")
        return response

