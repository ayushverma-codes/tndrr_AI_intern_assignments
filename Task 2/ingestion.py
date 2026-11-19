import json
import re
from functools import lru_cache
import numpy as np
import fitz
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, PDF, ARTIFACTS
@lru_cache(maxsize=1)
def get_model():
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    model.max_seq_length = 256
    return model
def clean_text(text):
    """
    Normalize whitespace without changing source content.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_document(pages, chunk_size, overlap):
    """
    Create overlapping token-based chunks.

    Each chunk stays within a single PDF page,
    preserving precise page-level citations.
    """
    assert 0 <= overlap < chunk_size

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        page_number = page["page"]
        text = clean_text(page["text"])

        if not text:
            continue

        token_ids = get_model().tokenizer.encode(
            text,
            add_special_tokens=False
        )

        for start in range(0, len(token_ids), step):
            end = min(start + chunk_size, len(token_ids))
            piece = token_ids[start:end]

            if not piece:
                continue

            chunk_text = get_model().tokenizer.decode(
                piece,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ).strip()

            if not chunk_text:
                continue

            chunks.append({
                "chunk_id": f"p{page_number}_t{start}",
                "page": page_number,
                "start_token": start,
                "end_token": end,
                "token_count": len(piece),
                "text": chunk_text
            })

            if end == len(token_ids):
                break

    return chunks
def load_index():
    if not PDF.is_file():
        raise RuntimeError(f'Missing document: {PDF}')
    chunks_path = ARTIFACTS / 'chunks_small.json'
    vectors_path = ARTIFACTS / 'embeddings_small.npy'
    ARTIFACTS.mkdir(exist_ok=True)
    if chunks_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    else:
        with fitz.open(PDF) as doc:
            pages = [{'page': i, 'text': page.get_text('text').strip()} for i, page in enumerate(doc, 1)]
        chunks = chunk_document(pages, 128, 24)
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding='utf-8')
    if vectors_path.exists():
        vectors = np.load(vectors_path, allow_pickle=False)
    else:
        vectors = get_model().encode([c['text'] for c in chunks], batch_size=32, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        np.save(vectors_path, vectors)
    if len(chunks) != 212 or vectors.shape != (212,384) or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1), 1, atol=1e-5):
        raise RuntimeError('Index does not match the notebook: expected 212 normalized 384-dimensional rows.')
    return chunks, vectors
