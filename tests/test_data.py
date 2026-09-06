"""Byte encoding/decoding round-trips (data.py)."""

from __future__ import annotations

from ringkospace.data import BYTE_OFFSET, decode, decode_char_boundary, encode_fast


def test_roundtrip_chinese() -> None:
    text = "你好，世界！Hello, 世界。\n"
    ids = encode_fast(text)
    assert (ids - BYTE_OFFSET >= 0).all() and (ids - BYTE_OFFSET < 256).all()
    assert decode(ids) == text


def test_roundtrip_ascii_no_drift() -> None:
    text = "abc 123 \t\n"
    ids = encode_fast(text)
    assert decode(ids) == text
    # ascii stays at one byte per char
    assert len(ids) == len(text.encode("utf-8"))


def test_decode_char_boundary_trims_half_codepoint() -> None:
    raw = "中文字符".encode("utf-8")[:7]  # cut inside the last char
    assert raw[-1] >= 0x80  # continuation byte => incomplete codepoint
    clean = decode_char_boundary(raw)
    assert clean.decode("utf-8") == "中文"  # dropped the partial char, kept prefix
