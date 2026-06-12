"""Request bodies for the segment endpoints."""
from __future__ import annotations

from pydantic import BaseModel


class SegmentCreate(BaseModel):
    song_index: int
    start: float
    end: float
    text: str
    lyric_start: int = -1
    lyric_end: int = -1


class SegmentUpdate(BaseModel):
    audio_filepath: str
    start: float
    end: float
    text: str
    lyric_start: int = -1
    lyric_end: int = -1


class SegmentDelete(BaseModel):
    audio_filepath: str


class LrcOffset(BaseModel):
    offset: float


class RejectedKey(BaseModel):
    key: str
