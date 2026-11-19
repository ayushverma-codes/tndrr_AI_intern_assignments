import os
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=True)
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
GROQ_MODEL = 'openai/gpt-oss-20b'
PDF = ROOT / 'data/nist_ai_rmf.pdf'
ARTIFACTS = ROOT / 'artifacts'
ABSTAIN = 'Insufficient information in the retrieved document.'
TOP_K, CANDIDATE_K, RRF_K = 5, 30, 60
MIN_COVERAGE = 0.45
def api_key():
    key = os.getenv('GROQ_API_KEY') or os.getenv('api_key')
    if not key or not key.strip():
        raise RuntimeError('Set GROQ_API_KEY in the project .env file before starting.')
    return key.strip()
