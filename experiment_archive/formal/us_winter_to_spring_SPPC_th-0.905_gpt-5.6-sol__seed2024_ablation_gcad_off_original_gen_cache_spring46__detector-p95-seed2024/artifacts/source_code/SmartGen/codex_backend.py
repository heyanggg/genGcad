from __future__ import annotations

import ast
import shutil
import subprocess
from pathlib import Path


class CodexGenerationError(RuntimeError):
    """Raised when Codex cannot return a usable SmartGen sequence."""


class CodexClient:
    """Run GPT-5.6 through the authenticated local Codex CLI."""

    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        executable: str = "codex",
        timeout: int = 900,
        reasoning_effort: str = "none",
        working_directory: str | Path | None = None,
    ):
        allowed_efforts = {"none", "low", "medium", "high", "xhigh", "max"}
        if reasoning_effort not in allowed_efforts:
            raise ValueError(f"reasoning_effort must be one of {sorted(allowed_efforts)}")
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"Codex CLI was not found: {executable}. Install Codex and log in before generation."
            )
        self.executable = resolved
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.working_directory = Path(working_directory or Path(__file__).resolve().parent)

    @property
    def generation_protocol(self) -> dict[str, object]:
        """Describe the effective, auditable Codex generation settings.

        The original SmartGen request used temperature=0, top_p=0, seed=2024 and
        max_tokens=8040. Codex CLI does not expose those request fields, so they
        must not be represented as active controls in an experiment manifest.
        """
        return {
            "transport": "codex_cli",
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "ephemeral": True,
            "ignore_user_config": True,
            "ignore_rules": True,
            "sandbox": "read-only",
            "original_smartgen_sampling_request": {
                "temperature": 0,
                "top_p": 0,
                "seed": 2024,
                "max_tokens": 8040,
            },
            "sampling_controls_applied": False,
            "sampling_controls_reason": (
                "Codex CLI does not recognize temperature, top_p, seed, or max_tokens "
                "as configuration fields"
            ),
        }

    def generate(self, prompt: str) -> str:
        command = [
            self.executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--model",
            self.model,
            "--config",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--cd",
            str(self.working_directory),
            "-",
        ]
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexGenerationError(
                f"Codex generation exceeded the {self.timeout}-second timeout."
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown Codex error"
            raise CodexGenerationError(f"Codex generation failed: {detail}")

        sequence_block = self.extract_sequence_block(completed.stdout)
        self.parse_sequence_block(sequence_block)
        return sequence_block

    @staticmethod
    def extract_sequence_block(response: str) -> str:
        response = response.strip()
        start = response.find("<seq")
        end = response.rfind("seq>")
        if start < 0 or end < start:
            raise CodexGenerationError("Codex response does not contain the required <seq ... seq> block.")
        return response[start : end + len("seq>")]

    @staticmethod
    def parse_sequence_block(sequence_block: str) -> list[list[str]]:
        content = sequence_block[len("<seq") : -len("seq>")].strip()
        try:
            sequences = ast.literal_eval(content)
        except (SyntaxError, ValueError) as exc:
            raise CodexGenerationError("The <seq ... seq> block is not a Python list.") from exc
        if not isinstance(sequences, list) or any(
            not isinstance(sequence, list)
            or not sequence
            or len(sequence) % 4 != 0
            or any(not isinstance(item, str) for item in sequence)
            for sequence in sequences
        ):
            raise CodexGenerationError(
                "Each generated sequence must be a non-empty list of text quadruplets."
            )
        return sequences

    @classmethod
    def is_valid_response(cls, response: object) -> bool:
        if not isinstance(response, str):
            return False
        try:
            sequence_block = cls.extract_sequence_block(response)
            cls.parse_sequence_block(sequence_block)
        except CodexGenerationError:
            return False
        return True
