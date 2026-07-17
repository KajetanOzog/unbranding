import json
import re


def extract_json(text):

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    candidates = re.findall(
        r"\{.*?\}",
        text,
        flags=re.DOTALL,
    )

    parsed = None

    for candidate in candidates:

        try:
            parsed = json.loads(candidate)

        except json.JSONDecodeError:
            continue

    return parsed

    return None