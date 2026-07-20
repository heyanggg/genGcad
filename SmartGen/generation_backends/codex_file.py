from __future__ import annotations

from pathlib import Path

from .conversion import convert_responses_to_smartgen
from .request_builder import export_requests
from .validation import validate_responses


class CodexFileBackend:
    """Offline file protocol; it contains no network client and requires no API key."""

    backend_type = "codex_file"
    generation_backend = "codex_agent_file"
    requires_api_key = False

    def export(self, output_dir: str | Path, **kwargs) -> list[dict]:
        return export_requests(output_dir, **kwargs)

    def validate(self, output_dir: str | Path, **kwargs) -> dict:
        directory = Path(output_dir)
        return validate_responses(
            directory / "generation_requests.jsonl",
            directory / "generation_responses_raw.jsonl",
            directory,
            **kwargs,
        )

    def convert(self, output_dir: str | Path, **kwargs) -> Path:
        directory = Path(output_dir)
        responses = directory / "generation_responses_selected.jsonl"
        if not responses.exists():
            responses = directory / "generation_responses_validated.jsonl"
        return convert_responses_to_smartgen(responses, directory, **kwargs)
