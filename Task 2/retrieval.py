import re
import math
from collections import Counter
import numpy as np
from ingestion import get_model, load_index
STOPWORDS = {
    "what", "which", "when", "where", "why", "how",
    "does", "do", "did", "is", "are", "was", "were",
    "the", "a", "an", "of", "in", "on", "for", "to",
    "and", "or", "with", "from", "by", "about",
    "can", "should", "would", "could", "its",
    "this", "that", "these", "those", "it",
    "be", "been", "being", "as", "at", "their"
}
def bm25_tokenize(text):
    """
    Simple lexical tokenizer.
    """
    return re.findall(
        r"[a-z0-9]+",
        text.lower()
    )

class BM25FromScratch:

    def __init__(
        self,
        documents,
        k1=1.5,
        b=0.75
    ):

        self.documents = documents
        self.k1 = k1
        self.b = b

        self.tokenized = [
            bm25_tokenize(doc)
            for doc in documents
        ]

        self.N = len(self.tokenized)

        self.doc_lengths = np.array(
            [len(doc) for doc in self.tokenized],
            dtype=np.float32
        )

        self.avgdl = float(
            np.mean(self.doc_lengths)
        )

        self.term_frequencies = [
            Counter(doc)
            for doc in self.tokenized
        ]

        document_frequencies = Counter()

        for tokens in self.tokenized:
            document_frequencies.update(
                set(tokens)
            )

        self.idf = {}

        for term, df in document_frequencies.items():

            self.idf[term] = math.log(
                1 + (
                    self.N - df + 0.5
                ) / (
                    df + 0.5
                )
            )

    def score(self, query):

        query_terms = bm25_tokenize(query)

        scores = np.zeros(
            self.N,
            dtype=np.float32
        )

        for term in query_terms:

            if term not in self.idf:
                continue

            idf = self.idf[term]

            for i, frequencies in enumerate(
                self.term_frequencies
            ):

                tf = frequencies.get(term, 0)

                if tf == 0:
                    continue

                denominator = (
                    tf
                    + self.k1 * (
                        1 - self.b
                        + self.b
                        * self.doc_lengths[i]
                        / max(self.avgdl, 1e-9)
                    )
                )

                scores[i] += (
                    idf
                    * tf
                    * (self.k1 + 1)
                    / denominator
                )

        return scores
def important_terms(text):
    words = re.findall(
        r"[a-z0-9]+",
        text.lower()
    )

    return set(
        word
        for word in words
        if len(word) >= 3
        and word not in STOPWORDS
    )

def evidence_relevance_check(
    question,
    retrieved,
    min_coverage=0.45
):
    """
    Check whether any retrieved passage contains
    enough important query terms.

    This is a lexical heuristic, not a semantic
    entailment or factual-grounding verifier.
    """

    query_terms = important_terms(question)

    if not query_terms:
        return {
            "relevant": False,
            "coverage": 0.0,
            "matched_terms": [],
            "query_terms": []
        }

    best_coverage = 0.0
    best_matches = set()

    for item in retrieved:

        passage_terms = important_terms(
            item["text"]
        )

        matched = query_terms & passage_terms

        coverage = (
            len(matched) / len(query_terms)
        )

        if coverage > best_coverage:
            best_coverage = coverage
            best_matches = matched

    return {
        "relevant": best_coverage >= min_coverage,
        "coverage": round(best_coverage, 4),
        "matched_terms": sorted(best_matches),
        "query_terms": sorted(query_terms)
    }
def split_evidence_sentences(text):
    """
    Split retrieved text into candidate evidence spans.

    Keep consecutive sentence pairs as well as individual
    sentences to reduce loss of context.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip()) >= 45
    ]

    spans = []

    for i, sentence in enumerate(sentences):
        spans.append(sentence)

        if i + 1 < len(sentences):
            spans.append(
                sentence + " " + sentences[i + 1]
            )

    return spans
class Retriever:
    def __init__(self):
        self.chunks, self.embeddings = load_index()
        self.model = get_model()
        self.bm25 = BM25FromScratch([c['text'] for c in self.chunks])
    def hybrid_retrieve(
        self, query,
        chunk_set="small",
        top_k=5,
        candidate_k=30,
        rrf_k=60
    ):
        """
        Combine dense cosine similarity and BM25 rankings
        using Reciprocal Rank Fusion.
        """

        chunks = self.chunks

        embeddings = (
            self.embeddings
        )

        # Dense scores
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        dense_scores = (
            embeddings @ query_embedding
        )

        # BM25 scores
        lexical_scores = (
            self.bm25.score(query)
        )

        n = len(chunks)

        candidate_k = min(
            candidate_k,
            n
        )

        dense_order = np.argsort(
            -dense_scores
        )[:candidate_k]

        bm25_order = np.argsort(
            -lexical_scores
        )[:candidate_k]

        # Reciprocal Rank Fusion
        fused_scores = {}

        for rank, idx in enumerate(
            dense_order,
            start=1
        ):

            idx = int(idx)

            fused_scores[idx] = (
                fused_scores.get(idx, 0.0)
                + 1.0 / (rrf_k + rank)
            )

        for rank, idx in enumerate(
            bm25_order,
            start=1
        ):

            idx = int(idx)

            fused_scores[idx] = (
                fused_scores.get(idx, 0.0)
                + 1.0 / (rrf_k + rank)
            )

        final_order = sorted(
            fused_scores,
            key=lambda idx: (
                -fused_scores[idx],
                idx
            )
        )[:top_k]

        results = []

        for rank, idx in enumerate(
            final_order,
            start=1
        ):

            chunk = chunks[idx]

            results.append({
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "score": float(
                    fused_scores[idx]
                ),
                "dense_score": float(
                    dense_scores[idx]
                ),
                "bm25_score": float(
                    lexical_scores[idx]
                ),
                "text": chunk["text"]
            })

        return results

    def select_question_evidence(
        self, question,
        chunk_set="small",
        candidate_k=15,
        top_n=5
    ):

        # Retrieve a larger candidate pool.
        retrieved = self.hybrid_retrieve(
            query=question,
            chunk_set=chunk_set,
            top_k=candidate_k,
            candidate_k=40,
            rrf_k=60
        )

        evidence = []
        seen = set()

        for item in retrieved:

            for span in split_evidence_sentences(
                item["text"]
            ):

                normalized = " ".join(
                    span.lower().split()
                )

                if normalized in seen:
                    continue

                seen.add(normalized)

                evidence.append({
                    "text": span,
                    "page": int(item["page"]),
                    "chunk_id": item["chunk_id"],
                    "retrieval_rank": int(item["rank"])
                })

        if not evidence:
            return []

        # Compare question and evidence semantically.
        query_vector = self.model.encode(
            [question],
            normalize_embeddings=True
        )[0]

        evidence_vectors = self.model.encode(
            [item["text"] for item in evidence],
            normalize_embeddings=True,
            batch_size=32
        )

        semantic_scores = (
            evidence_vectors @ query_vector
        )

        query_terms = important_terms(question)

        for i, item in enumerate(evidence):

            span_terms = important_terms(
                item["text"]
            )

            lexical_score = (
                len(query_terms & span_terms)
                / max(len(query_terms), 1)
            )

            # Prefer semantic relevance while also
            # rewarding exact terminology.
            item["score"] = float(
                0.75 * semantic_scores[i]
                + 0.25 * lexical_score
            )

        evidence.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        # Avoid returning multiple overlapping spans
        # from the same chunk where possible.
        selected = []
        used_chunks = set()

        for item in evidence:

            if item["chunk_id"] in used_chunks:
                continue

            selected.append(item)
            used_chunks.add(item["chunk_id"])

            if len(selected) >= top_n:
                break

        return selected

    def select_complete_evidence(
        self, question,
        chunk_set="small",
        candidate_k=15,
        top_n=3,
        min_words=12
    ):
        """
        Retrieve and select evidence while filtering out
        very short fragments.
        """

        # Request more evidence than needed so that
        # filtering does not leave too few candidates.
        candidates = self.select_question_evidence(
            question=question,
            chunk_set=chunk_set,
            candidate_k=candidate_k,
            top_n=15
        )

        selected = []

        for item in candidates:

            words = re.findall(
                r"\b\w+\b",
                item["text"]
            )

            if len(words) < min_words:
                continue

            selected.append(item)

            if len(selected) == top_n:
                break

        return selected
