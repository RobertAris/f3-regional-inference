"""Resolve local packet locations and hash exact release bytes."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
