"""Envelope protocol v1 (PLAN §5): {v, id, type, ts, payload}.

stdout carries the handshake JSON only; every WS message uses this
envelope. ``id`` correlates request/response (WS is full-duplex and
responses may interleave), ``v`` gates protocol evolution.
"""
from __future__ import annotations

import json
import time


def envelope(msg_id: str, mtype: str, payload) -> str:
    return json.dumps(
        {
            "v": 1,
            "id": msg_id,
            "type": mtype,
            "ts": int(time.time() * 1000),
            "payload": payload,
        }
    )


def ok(msg_id: str, payload) -> str:
    return envelope(msg_id, "res", payload)


def err(msg_id: str, error: str) -> str:
    return envelope(msg_id, "res", {"error": error})


def parse(raw: str | bytes) -> tuple[str, str, dict]:
    """Return (id, type, payload); raise ValueError on malformed input."""
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"bad json: {e}") from e
    if not isinstance(msg, dict):
        raise TypeError("message must be a json object")
    msg_id = str(msg.get("id", "m-0"))
    mtype = msg.get("type")
    if not isinstance(mtype, str):
        raise TypeError("missing type")
    payload = msg.get("payload")
    if payload is not None and not isinstance(payload, dict):
        raise TypeError("payload must be an object when present")
    return msg_id, mtype, payload or {}
