"""
Voice Recognition & Speaker Identification Package.
"""

from .recognizer import (
    SpeakerDatabase,
    default_db,
    extract_voice_fingerprint,
)
from .sentences import (
    PHONETIC_SENTENCES,
    get_random_sentence,
    get_all_sentences,
)

__all__ = [
    "SpeakerDatabase",
    "default_db",
    "extract_voice_fingerprint",
    "PHONETIC_SENTENCES",
    "get_random_sentence",
    "get_all_sentences",
]

