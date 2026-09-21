"""Robust JSON -> pydantic parsing for model output."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ailura.llm.base import StructuredOutputError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_block(text: str) -> str | None:
    """Return the first JSON object found in ``text`` (fenced or bare), else None."""
    m = _FENCE.search(text)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("{"):
            return candidate
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]
    return None


def parse_into(schema: type[T], text: str) -> T:
    """Parse ``text`` into ``schema``; tolerant of fences and surrounding prose."""
    candidates = [text.strip()]
    block = extract_json_block(text)
    if block and block != candidates[0]:
        candidates.append(block)
    last_err: Exception | None = None
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            last_err = e
    raise StructuredOutputError(f"could not parse {schema.__name__}: {last_err}")


def schema_instructions(schema: type[BaseModel]) -> str:
    """Prompt fragment describing the required JSON schema (for providers without native structured output)."""
    return (
        "Respond with a single JSON object and nothing else. It must validate against this JSON schema:\n"
        + json.dumps(schema.model_json_schema(), indent=None, sort_keys=True)
    )
