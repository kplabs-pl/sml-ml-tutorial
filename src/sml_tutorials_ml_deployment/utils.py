from typing import Any, Mapping


def add_prefix_to_dict_keys(d: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": d[key] for key in d}
