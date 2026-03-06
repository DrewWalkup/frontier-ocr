from __future__ import annotations

import re

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_MARKDOWN_CODEBLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)


def markdown_to_plain_text(markdown: str) -> str:
    """
    Best-effort conversion of markdown into readable plain text.

    This is intentionally simple and predictable.
    For perfect fidelity, consumers should parse structured_result instead.
    """
    text = markdown

    # Remove code blocks entirely
    text = _MARKDOWN_CODEBLOCK_RE.sub("", text)

    # Convert links to just their visible text
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)

    # Remove images (keep alt text if present)
    text = _MARKDOWN_IMAGE_RE.sub(r"\1", text)

    # Strip some common markdown formatting characters
    text = text.replace("**", "").replace("__", "").replace("*", "").replace("`", "")

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text
