"""Map canonical PX4 topic names onto a vehicle's live namespace."""


def live_px4_topic(canonical_topic: str, namespace: str = "") -> str:
    """Return the live DDS name while keeping bag names vehicle-independent."""
    if not canonical_topic.startswith("/fmu/"):
        raise ValueError(f"not a canonical PX4 topic: {canonical_topic}")
    parts = [part for part in namespace.split("/") if part]
    prefix = f"/{'/'.join(parts)}" if parts else ""
    return f"{prefix}{canonical_topic}"
