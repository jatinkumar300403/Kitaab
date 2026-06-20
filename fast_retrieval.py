import re
from typing import Dict, List

import pandas as pd


def load_corpus(path: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            isbn, text = parts
            result[isbn] = text
    return result


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def top_matches(query: str, corpus: Dict[str, str], top_k: int = 50) -> List[str]:
    if not query:
        return []
    query_words = set(normalize_text(query).split())
    if not query_words:
        return []

    scores: Dict[str, int] = {}
    for isbn, text in corpus.items():
        text_words = set(normalize_text(text).split())
        scores[isbn] = len(query_words & text_words)

    top_isbns = sorted(scores, key=lambda isbn: (-scores[isbn], isbn))[:top_k]
    return [isbn for isbn in top_isbns if scores[isbn] > 0]
