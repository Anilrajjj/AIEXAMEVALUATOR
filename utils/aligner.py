from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.12


def align_segments_to_questions(questions: List[str], segments: List[str]) -> List[str]:
    """Align extracted segments to question texts using TF-IDF cosine similarity.

    Returns a list of segments ordered to match the questions list. If no good
    match is found for a question, an empty string is returned for that slot.
    """
    if not questions:
        return []

    # Normalize empty inputs
    segments = segments or []

    # Build corpus to ensure consistent vocabulary
    corpus = list(questions) + (segments if segments else [""])

    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    try:
        vectorizer.fit(corpus)
    except Exception:
        # Fallback: return empty assignments
        return ["" for _ in questions]

    q_vecs = vectorizer.transform(questions)
    if segments:
        s_vecs = vectorizer.transform(segments)
    else:
        s_vecs = None

    assignments = ["" for _ in questions]
    if s_vecs is None or s_vecs.shape[0] == 0:
        return assignments

    sim = cosine_similarity(q_vecs, s_vecs)

    used = set()
    # Greedy assignment: for each question choose best unused segment
    for qi in range(sim.shape[0]):
        row = sim[qi]
        if row.size == 0:
            continue
        # choose the highest similarity among unused segments
        candidates = sorted([(float(row[j]), j) for j in range(row.size)], reverse=True)
        chosen = None
        for val, j in candidates:
            if j in used:
                continue
            if val >= SIMILARITY_THRESHOLD:
                chosen = j
                used.add(j)
                break
        if chosen is not None:
            assignments[qi] = segments[chosen]
        else:
            assignments[qi] = ""

    return assignments
