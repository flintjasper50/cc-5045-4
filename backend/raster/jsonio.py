"""JSON-safe serialisation of render payloads.

Python's ``json`` emits bare ``Infinity`` / ``NaN`` tokens, which are not valid
JSON and break browser ``JSON.parse``.  Background depth is legitimately +inf,
so instead of hiding it we serialise non-finite floats as explicit strings
(``"Infinity"``, ``"-Infinity"``, ``"NaN"``) -- the report then states exactly
where a non-finite value sits.
"""
import math


def sanitize(value):
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (bytearray, bytes)):
        return list(value)
    return value


def dumps(obj, **kwargs):
    import json
    return json.dumps(sanitize(obj), **kwargs)
