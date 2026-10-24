"""Use committed shared history plus the current round's private input/tail."""

import copy
import json
from collections import Counter

from astrbot.core.agent.message import bind_checkpoint_messages


def key(message):
    value = copy.deepcopy(message)
    content = value.get("content")
    if isinstance(content, list) and all(p.get("type") == "text" for p in content):
        value["content"] = "".join(
            p.get("text", "") for p in content if not p.get("_no_save")
        )

    def clean(v):
        if isinstance(v, dict):
            return {
                k: clean(x)
                for k, x in v.items()
                if k != "_no_save" and x not in (None, "", [], {})
            }
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    return json.dumps(clean(value), sort_keys=True, ensure_ascii=False)


def without_inputs(history, inputs):
    """Remove this round's admitted inputs once, including identical messages."""
    counts = Counter(key(v) for v in inputs)
    kept = []
    # Inputs were appended at admission; prefer the latest equal occurrence.
    for message in reversed(history):
        ident = key(message)
        if counts[ident]:
            counts[ident] -= 1
        else:
            kept.append(message)
    return list(reversed(kept))


def bind(history):
    return bind_checkpoint_messages(copy.deepcopy(history))
